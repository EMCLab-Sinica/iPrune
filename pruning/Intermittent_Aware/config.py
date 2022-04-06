config = {
    'LeNet_5': [ # NCHW
        {
            'input': [1,1,28,28],
            'filter': [6,1,5,5],
            'output': [1,6,28,28],
            'tile': {
                'input': [1,1,28,2],#'input': [1,1,28,3],
                'weight': [6,1,1,1],
                'output': [1,6,28,2],#'output': [1,8,28,3],
            },
            'group': [6, 1], # [n_filter, n_channel]
            'pads': [2, 2, 2, 2],
            'stride': 1
        },
        {
            'input': [1,6,14,14],
            'filter': [16,6,5,5],
            'output': [1,16,10,10],
            'tile': {
                'input': [1,6,14,4],#'input': [1,4,14,6],
                'weight': [4,6,1,1],
                'output': [1,4,10,4],#'output': [1,4,14,6],
            },
            'group': [4, 6],
            'pads': [0, 0, 0, 0],
            'stride': 1
        },
        {
            'input': [1,16*5*5,1,1],
            'filter': [120,16*5*5,1,1],
            'output': [1,120,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 16],
            'stride': 1
        },
        {
            'input': [1,120,1,1],
            'filter': [84,120,1,1],
            'output': [1,84,1,1],
            'tile': {
                'input': [1,24,1,1],
                'weight': [2,24,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 24],
            'stride': 1
        },
        {
            'input': [1,84,1,1],
            'filter': [10,84,1,1],
            'output': [1,10,1,1],
            'tile': {
                'input': [1,12,1,1],
                'weight': [2,12,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 12],
            'stride': 1
        }
    ],
    'mnist': [
        {
            'input': [1,1,28,28],
            'filter': [8,1,5,5],
            'output': [1,8,28,28],
            'tile': {
                'input': [1,1,28,3],
                'weight': [4,1,1,1],
                'output': [1,4,28,3],
            },
            'group': [4, 1],
            'pads': [2, 2, 2, 2],
            'stride': 1
        },
        {
            'input': [1,8,14,14],
            'filter': [16,8,5,5],
            'output': [1,16,14,14],
            'tile': {
                'input': [1,4,14,6],
                'weight': [4,4,1,1],
                'output': [1,4,14,6],
            },
            'group': [4, 4],
            'pads': [2, 2, 2, 2],
            'stride': 1
        },
        {
            'input': [1,16*4*4,1,1],
            'filter': [10,16*4*4,1,1],
            'output': [1,10,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 16],
            'stride': 1
        }
    ],
    'HAR': [
        {
            'input': [1,9,1,128],
            'filter': [18,9,1,2],
            'output': [1,18,1,128],
            'tile': {
                'input': [1,9,1,32],
                'weight': [6,9,1,1],
                'output': [1,6,1,32],
            },
            'group': [6, 9],
            'pads': [0, 1, 0, 0],
            'stride': 1
        },
        {
            'input': [1,18,1,64],
            'filter': [36,18,1,2],
            'output': [1,36,1,64],
            'tile': {
                'input': [1,18,1,32],
                'weight': [6,18,1,1],
                'output': [1,6,1,32],
            },
            'group': [6, 18],
            'pads': [0, 1, 0, 0],
            'stride': 1
        },
        {
            'input': [1,36,1,32],
            'filter': [72,36,1,2],
            'output': [1,72,1,32],
            'tile': {
                'input': [1,36,1,30],
                'weight': [6,36,1,1],
                'output': [1,6,1,30],
            },
            'group': [6, 36],
            'pads': [0, 1, 0, 0],
            'stride': 1
        },
        {
            'input': [1,16*72,1,1],
            'filter': [6,16*72,1,1],
            'output': [1,6,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 16],
            'stride': 1
        },
    ],
    'KWS': [
        {
            'input': [1,25*10,1,1],
            'filter': [144,250,1,1],
            'output': [1,144,1,1],
            'tile': {
                'input': [1,10,1,1],
                'weight': [24,10,1,1],
                'output': [1,24,1,1],
            },
            'group': [12, 10],
            'stride': 1
        },
        {
            'input': [1,144,1,1],
            'filter': [144,144,1,1],
            'output': [1,144,1,1],
            'tile': {
                'input': [1,6,1,1],
                'weight': [24,6,1,1],
                'output': [1,24,1,1],
            },
            'group': [12, 6],
            'stride': 1
        },
        {
            'input': [1,144,1,1],
            'filter': [144,144,1,1],
            'output': [1,144,1,1],
            'tile': {
                'input': [1,6,1,1],
                'weight': [24,6,1,1],
                'output': [1,24,1,1],
            },
            'group': [12, 6],
            'stride': 1
        },
        {
            'input': [1,144,1,1],
            'filter': [12,144,1,1],
            'output': [1,12,1,1],
            'tile': {
                'input': [1,6,1,1],
                'weight': [12,6,1,1],
                'output': [1,12,1,1],
            },
            'group': [12, 6],
            'stride': 1
        },
    ],
    'SqueezeNet': [
        #0
        {
            'input': [1,3,32,32],
            'filter': [64,3,3,3],
            'output': [1,64,15,15],
            'tile': {
                'input': [1,3,32,2],
                'weight': [2,3,1,1],
                'output': [1,2,15,1],
            },
            'group': [2, 3],
            'pads': [1,1,0,0],
            'stride': 2
        },
        #1
        {
            'input': [1,64,7,7],
            'filter': [16,64,1,1],
            'output': [1,16,7,7],
            'tile': {
                'input': [1,64,7,1],
                'weight': [16,64,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 64],
            'pads': [0,0,0,0],
            'stride': 1
        },
        #2
        {
            'input': [1,16,7,7],
            'filter': [64,16,1,1],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,1],
                'weight': [16,16,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 16],
            'pads': [0,0,0,0],
            'stride': 1
        },
        #3
        {
            'input': [1,16,7,7],
            'filter': [64,16,3,3],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,1],
                'weight': [16,16,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 16],
            'pads': [1,1,1,1],
            'stride': 1
        },
        #4
        {
            'input': [1,128,7,7],
            'filter': [16,128,1,1],
            'output': [1,16,7,7],
            'tile': {
                'input': [1,16,7,1],
                'weight': [16,16,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 16],
            'pads': [0,0,0,0],
            'stride': 1
        },
        #5
        {
            'input': [1,16,7,7],
            'filter': [64,16,1,1],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,1],
                'weight': [16,16,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 16],
            'pads': [0,0,0,0],
            'stride': 1
        },
        #6
        {
            'input': [1,16,7,7],
            'filter': [64,16,3,3],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,1],
                'weight': [16,16,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 16],
            'pads': [1,1,1,1],
            'stride': 1
        },
        #7
        {
            'input': [1,128,7,7],
            'filter': [32,128,1,1],
            'output': [1,32,7,7],
            'tile': {
                'input': [1,64,7,1],
                'weight': [16,64,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 64],
            'pads': [0,0,0,0],
            'stride': 1
        },
        #8
        {
            'input': [1,32,7,7],
            'filter': [128,32,1,1],
            'output': [1,128,7,7],
            'tile': {
                'input': [1,32,7,1],
                'weight': [16,32,1,1],
                'output': [1,16,7,1],
            },
            'group': [16, 32],
            'pads': [0,0,0,0],
            'stride': 1
        },
        #9
        {
            'input': [1,32,7,7],
            'filter': [128,32,3,3],
            'output': [1,128,7,7],
            'tile': {
                'input': [1,32,7,1],
                'weight': [8,32,1,1],
                'output': [1,8,7,1],
            },
            'group': [8, 32],
            'pads': [1,1,1,1],
            'stride': 1
        },
        #10
        {
            'input': [1,256,7,7],
            'filter': [10,256,1,1],
            'output': [1,10,7,7],
            'tile': {
                'input': [1,128,7,1],
                'weight': [2,128,1,1],
                'output': [1,2,7,1],
            },
            'group': [2, 128],
            'pads': [0,0,0,0],
            'stride': 1
        }
    ]
}
