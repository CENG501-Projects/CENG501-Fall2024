import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import os
import math

from tqdm.auto import tqdm


class Args:
    def __init__(self, **kwargs):
        # Optimizer
        self.lr = kwargs.get('lr', 0.01)
        self.weight_decay = kwargs.get('weight_decay', 0.01)
        # Scheduler
        # self.patience = kwargs.get('patience', 10)
        self.epochs = kwargs.get('epochs', 100)
        # Dataset
        self.dataset = kwargs.get('dataset', 'cifar10')
        self.grayscale = kwargs.get('grayscale', False)
        # Dataloader
        self.num_workers = kwargs.get('num_workers', 2)
        self.batch_size = kwargs.get('batch_size', 64)
        # Model
        self.n_layers = kwargs.get('n_layers', 4)
        self.d_model = kwargs.get('d_model', 512)
        self.d_inner = kwargs.get('d_inner', 2048)
        self.d_k = kwargs.get('d_k', 64)
        self.d_v = kwargs.get('d_v', 64)
        self.n_heads = kwargs.get('n_heads', 8)
        self.dropout = kwargs.get('dropout', 0.0)
        self.prenorm = kwargs.get('prenorm', False)
        # General
        self.resume = kwargs.get('resume', False)
        # Kernel size
        self.kernel_size = kwargs.get('kernel_size', 5)
        # B parameter
        self.b = kwargs.get('b', 0.001)
        # task
        self.task = kwargs.get('task', 'image')
        
args = Args(n_layers=6, lr = 1e-3, batch_size=10, kernel_size=3, epochs=2, weight_decay= 0.01, dataset='mnist', num_workers=2)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print(f'==> Preparing {args.dataset} data..')

def split_train_val(train, val_split):
    train_len = int(len(train) * (1.0-val_split))
    train, val = torch.utils.data.random_split(
        train,
        (train_len, len(train) - train_len),
        generator=torch.Generator().manual_seed(42),
    )
    return train, val

if args.dataset == 'cifar10':

    if args.grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=122.6 / 255.0, std=61.0 / 255.0),
            transforms.Lambda(lambda x: x.view(1, 1024).t())
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(3, 1024).t())
        ])

    # S4 is trained on sequences with no data augmentation!
    transform_train = transform_test = transform

    trainset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.CIFAR10(
        root='./data/cifar/', train=False, download=True, transform=transform_test)

    d_input = 3 if not args.grayscale else 1
    d_output = 10

elif args.dataset == 'mnist':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(1, 784).t())
    ])
    transform_train = transform_test = transform

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_train)
    trainset, _ = split_train_val(trainset, val_split=0.1)

    valset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_test)
    _, valset = split_train_val(valset, val_split=0.1)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_test)

    d_input = 1
    d_output = 10
else: raise NotImplementedError

# Dataloaders
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valloader = torch.utils.data.DataLoader(
    valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    


# alibi matrix taken from https://github.com/jaketae/alibi/blob/main/alibi/attention.py
def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class LaSAttention(nn.Module):
    ''' Local and Smooth (LaS) Attention '''
    def __init__(self, d_k, alpha_c, kernel_size):
        super().__init__()
        self.d_k = d_k
        self.alpha_c = alpha_c
        self.kernel_size = kernel_size
        self.padding = 2

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.d_k, k.transpose(2, 3))


        _, _, seq_len, _ = attn.shape

        d_l = get_relative_positions(seq_len=seq_len).to(device)

        eld_matrix = (self.alpha_c*(-1)*d_l).to(device)

        eld_attn = torch.mul(eld_matrix, attn)

        if(self.kernel_size == 3):
            self.padding = 1
        else:
            self.padding = (self.kernel_size - 1) // 2


        eld_attn = eld_attn.view(-1, eld_attn.size(2), eld_attn.size(3)) #flag
        eld_attn = F.avg_pool1d(input=F.softmax(eld_attn, dim=-1),
                                kernel_size=self.kernel_size,
                                stride=1, # we assumed the value of stride
                                padding=self.padding) # padding is adjusted to preserve shape

        eld_attn = eld_attn.view(q.size(0), q.size(1), eld_attn.size(1), eld_attn.size(2))

        min_seq_len = min(eld_attn.size(2), v.size(2))
        eld_attn = eld_attn[:, :, :min_seq_len, :]
        v = v[:, :, :min_seq_len, :]


        output = torch.matmul(eld_attn, v)

        return output, eld_attn

        """if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)"""
            
            
            
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v,  b, current_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.current_head = current_head
        self.b = b
        if (current_head==0):
          self.alpha_c = 0
        else:
          self.alpha_c = (-1)*math.log(b/(n_head-1) * current_head)

        self.attention = LaSAttention(d_k=d_k ** 0.5, alpha_c=self.alpha_c, kernel_size=args.kernel_size)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        #print("before")
        #print(q.shape, residual.shape)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))

        q = q.unsqueeze(2)
        #print("after")
        #print(q.shape, residual.shape)
        #print("\n")
        if len(residual.shape) == 3:
           residual = residual.unsqueeze(2)

        q += residual

        q = self.layer_norm(q)

        return q, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, b, current_head, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, b, current_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, b, current_head, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, b, current_head, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, b, current_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn



def get_pad_mask(seq, pad_idx):
    if pad_idx is None:
        return torch.ones_like(seq, dtype=torch.bool).unsqueeze(-2)  # Return all True mask
    else:
        return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, b, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,  b, current_head, dropout=dropout) #b flag
            for current_head in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        src_seq = (src_seq * 255).long() #flag

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        #enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.dropout(enc_output) #flag
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, b, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, b, current_head, dropout=dropout)
            for current_head in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        #dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.dropout(dec_output)
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, b=0.001, #b flag
            n_position=200, trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):

        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, b=b, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, b=b, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit.view(-1, seq_logit.size(2))



# Model
print('==> Building model..')
model = Transformer(
    n_src_vocab=256,             # Input pixel values (0-255)
    n_trg_vocab=10,              # Output classes (digits 0-9)
    src_pad_idx=None,            # Not used for Sequential MNIST
    trg_pad_idx=None,            # Not used for Sequential MNIST
    d_word_vec=args.d_model,              # Embedding size
    d_model=args.d_model,                 # Embedding dimension
    d_inner=args.d_inner,                 # Feedforward inner dimension
    n_layers=args.n_layers,                  # Number of layers
    n_head=args.n_heads,                    # Number of attention heads
    d_k=args.d_k,                      # Dimension of attention key
    d_v=args.d_v,                      # Dimension of attention value
    dropout=args.dropout,                 # Dropout rate
    b=args.b, #b flag
    n_position=784,              # Sequence length
    trg_emb_prj_weight_sharing=False,  # Disable weight sharing
    emb_src_trg_weight_sharing=False   # Disable weight sharing
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True
    
    
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def setup_optimizer(model, lr, weight_decay, epochs):
    """
    S4 requires a specific optimizer setup.

    The S4 layer (A, B, C, dt) parameters typically
    require a smaller learning rate (typically 0.001), with no weight decay.

    The rest of the model can be trained with a higher learning rate (e.g. 0.004, 0.01)
    and weight decay (if desired).
    """

    # All parameters in the model
    all_parameters = list(model.parameters())

    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer(
    model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs
)



import matplotlib.pyplot as plt

val_loss_lst = []
train_loss_lst = []




# Training
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Create target sequence
        trg_seq = targets.unsqueeze(1).repeat(1, inputs.size(1)) #flag

        optimizer.zero_grad()
        #outputs = model(inputs)
        #loss = criterion(outputs, targets)
        outputs = model(inputs, trg_seq) #flag
        trg_seq = trg_seq.flatten()

        outputs = outputs.view(trg_seq.size(0), -1, 10).mean(dim=1)  # Average across sequence
        # print("Outputs shape:", outputs.shape)
        # print("Trg_seq shape:", trg_seq.shape)
        loss = criterion(outputs, trg_seq)
        train_loss_lst.append(loss.item())
        #print("exited criterion")
        #loss = criterion(outputs, targets)
        #loss = criterion(outputs, trg_seq.view(-1, 10).repeat(1, 10)[:, targets])  # Correct loss calculation #flag
        loss.backward()
        #print("passed backward")
        optimizer.step()
        #print("passed optimizer step")

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += trg_seq.size(0)
        correct += predicted.eq(trg_seq).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total)
        )


def eval(epoch, dataloader, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            #outputs = model(inputs)
            #loss = criterion(outputs, targets)

            trg_seq = targets.unsqueeze(1).repeat(1, inputs.size(1)) #flag

            outputs = model(inputs, trg_seq)  #flag
            trg_seq = trg_seq.flatten()

            outputs = outputs.view(trg_seq.size(0), -1, 10).mean(dim=1)  # Average across sequence
            loss = criterion(outputs, trg_seq)

            if checkpoint:
              val_loss_lst.append(loss.item())

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            '''
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )
            '''
            # You might need to adjust how you calculate 'correct'
            # based on the output sequence and your task
            # For example, you could average the predictions across the sequence
            #below is flagged
            predicted_class = predicted.view(trg_seq.size(0), -1).float().mean(dim=1).round().long()
            total += trg_seq.size(0)
            correct += predicted_class.eq(trg_seq).sum().item()
            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)

                )


    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            # torch.save(state, './checkpoint/ckpt.pth')
            torch.save(state, '/content/drive/MyDrive/ceng501/codes/LARGE_LaS_checkpoint/ckpt.pth')

            best_acc = acc

        return acc
        
        
        
        
pbar = tqdm(range(start_epoch, args.epochs))
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    train()
    val_acc = eval(epoch, valloader, checkpoint=True)
    eval(epoch, testloader)
    scheduler.step()
    # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")
    
    
    
plt.plot(val_loss_lst)
plt.title("Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")



plt.plot(train_loss_lst)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
