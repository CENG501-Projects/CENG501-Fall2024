import torch

def autoGen_acrobotDynamics(q1, q2, dq1, dq2, m1, m2, g, l1, l2):
    """
    Auto-generated function for Acrobot dynamics.
    Compatible with PyTorch autograd for gradient computation.
    Returns the mass matrix (D), gravitational forces (G), and input matrix (B).
    """
    t2 = torch.cos(q1)
    t3 = l1 ** 2
    t4 = torch.sin(q1)
    t5 = torch.cos(q2)
    t6 = l1 * t2
    t7 = l2 * t5
    t8 = t6 + t7
    t9 = torch.sin(q2)
    t10 = l1 * t4
    t11 = l2 * t9
    t12 = t10 + t11
    t13 = l2 ** 2

    # Mass matrix D
    D = torch.stack([
        torch.stack([-m1 * t2 ** 2 * t3 - m1 * t3 * t4 ** 2 - l1 * m2 * t2 * t8 - l1 * m2 * t4 * t12,
                     -l1 * l2 * m2 * t2 * t5 - l1 * l2 * m2 * t4 * t9]),
        torch.stack([-l2 * m2 * t5 * t8 - l2 * m2 * t9 * t12,
                     -m2 * t5 ** 2 * t13 - m2 * t9 ** 2 * t13])
    ])

    # Gravitational vector G
    t14 = dq1 ** 2
    t15 = dq2 ** 2
    t16 = l1 * t2 * t14
    t17 = l2 * t5 * t15
    t18 = t16 + t17
    t19 = l1 * t4 * t14
    t20 = l2 * t9 * t15
    t21 = t19 + t20
    G = torch.stack([
        -g * m2 * t12 + m2 * t8 * t21 - m2 * t12 * t18 - g * l1 * m1 * t4,
        -g * l2 * m2 * t9 + l2 * m2 * t5 * t21 - l2 * m2 * t9 * t18
    ])

    # Input matrix B
    B = torch.tensor([0.0, -1.0], dtype=torch.float32)  # This is constant, no grad required.

    return D, G, B


# Example usage
q1 = torch.tensor(0.5, requires_grad=True)
q2 = torch.tensor(1.0, requires_grad=True)
dq1 = torch.tensor(0.2, requires_grad=True)
dq2 = torch.tensor(0.3, requires_grad=True)
m1 = torch.tensor(1.0)
m2 = torch.tensor(1.0)
g = torch.tensor(9.81)
l1 = torch.tensor(1.0)
l2 = torch.tensor(1.0)

D, G, B = autoGen_acrobotDynamics(q1, q2, dq1, dq2, m1, m2, g, l1, l2)

# Compute gradients (example)
loss = D.norm() + G.norm() + B.norm()  # Arbitrary loss function
loss.backward()  # Compute gradients
print(q1.grad)  # Gradient of the loss with respect to q1
print(q2.grad)  # Gradient of the loss with respect to q2
