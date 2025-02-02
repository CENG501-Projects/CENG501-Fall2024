# Define instance classes
instance_classes = {"road":0,
"sidewalk":1,
"parking":2,
"rail track":3,
"person":4,
"rider":5,
"car":6,
"truck":7,
"bus":8,
"on rails":9,
"motorcycle":10,
"bicycle":11,
"caravan":12,
"trailer":13,
"building":14,
"wall":15,
"fence":16,
"guard rail":17,
"bridge":18,
"tunnel":19,
"pole":20,
"pole group":21,
"traffic sign":22,
"traffic light":23,
"vegetation":24,
"terrain":25,
"sky":26,
"ground":27,
"dynamic":28,
"static":29}

# Categories: 'flat', 'construction', 'nature', 'vehicle', 'sky', 'object', 'human', 'void'
category_map = {
    'flat': 0,           
    'construction': 1,   
    'nature': 2,         
    'vehicle': 3,        
    'sky': 4,            
    'object': 5,         
    'human': 6,          
    'void': 7            
}

# Map instance classes to 8 categories
# Based on cityscapes webpage definition 
class_to_category = {"road":'flat',
"sidewalk":'flat',
"parking":'flat',
"rail track":'flat',
"person":"human",
"rider":"human",
"car":"vehicle",
"truck":"vehicle",
"bus":"vehicle",
"on rails":"vehicle",
"motorcycle":"vehicle",
"bicycle":"vehicle",
"caravan":"vehicle",
"trailer":"vehicle",
"building":"construction",
"wall":"construction",
"fence":"construction",
"guard rail":"construction",
"bridge":"construction",
"tunnel":"construction",
"pole":"object",
"pole group":"object",
"traffic sign":"object",
"traffic light":"object",
"vegetation":"nature",
"terrain":"nature",
"sky":"sky",
"ground":"void",
"dynamic":"void",
"static":"void"}



