config = {
    'LeNet_5': [ # NCHW
        {
            'input': [1,1,28,28],
            'filter': [8,1,5,5],
            'output': [1,8,28,28],
            'tile': {
                'input': [1,1,28,1],#'input': [1,1,28,3],
                'weight': [8,1,1,1],
                'output': [1,8,28,1],#'output': [1,8,28,3],
            },
            'group': [8, 1], # [n_filter, n_channel]
            'pads': [2, 2, 2, 2],
            'stride': 1
        },
        {
            'input': [1,8,14,14],
            'filter': [16,8,5,5],
            'output': [1,4,14,14],
            'tile': {
                'input': [1,4,14,6],#'input': [1,4,14,6],
                'weight': [4,4,1,1],
                'output': [1,4,14,6],#'output': [1,4,14,6],
            },
            'group': [4, 4],
            'pads': [2, 2, 2, 2],
            'stride': 1
        },
        {
            'input': [1,16*4*4,1,1],
            'filter': [256,16*4*4,1,1],
            'output': [1,256,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 16],
            'stride': 1
        },
        {
            'input': [1,256,1,1],
            'filter': [10,256,1,1],
            'output': [1,10,1,1],
            'tile': {
                'input': [1,256,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 16],
            'stride': 1
        }
    ],
    'mnist': [
        {
            'input': [1,1,28,28],
            'filter': [32,1,3,3],
            'output': [1,32,26,26],
            'tile': {
                'input': [1,1,28,2],
                'weight': [8,1,1,1],
                'output': [1,8,26,2],
            },
            'group': [8, 1],
            'pads': [0, 0, 0, 0],
            'stride': 1
        },
        {
            'input': [1,1,13,13],
            'filter': [64,32,3,3],
            'output': [1,64,11,11],
            'tile': {
                'input': [1,8,13,8],
                'weight': [4,8,1,1],
                'output': [1,4,11,8],
            },
            'group': [4, 8],
            'pads': [0, 0, 0, 0],
            'stride': 1
        },
        {
            'input': [1,64*5*5,1,1],
            'filter': [128,64*5*5,1,1],
            'output': [1,128,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2, 16],
            'stride': 1
        },
        {
            'input': [1,128,1,1],
            'filter': [10,128,1,1],
            'output': [1,10,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [2,16,1,1],
                'output': [1,2,1,1],
            },
            'group': [2,16],
            'stride': 1
        },
    ],
    'SqueezeNet': [
        #0
        {
            'input': [32,32,3],
            'filter': [3,3,3,64],
            'output': [16,16,64],
            'tile': {
                'input': [8,3,3],
                'weight': [3,3,3,64]
            },
            'group': [2, 1],
            'stride': 2
        },
        #1
        {
            'input': [7,7,64],
            'filter': [1,1,64,16],
            'output': [7,7,16],
            'tile': {
                'input': [7,3,64],
                'weight': [1,1,64,16]
            },
            'group': [2, 1],
            'stride': 1
        },
        #2
        {
            'input': [7,7,16],
            'filter': [1,1,16,64],
            'output': [7,7,64],
            'tile': {
                'input': [7,3,16],
                'weight': [1,1,16,64]
            },
            'group': [2, 1],
            'stride': 1
        },
        #3
        {
            'input': [7,7,16],
            'filter': [3,3,16,64],
            'output': [7,7,64],
            'tile': {
                'input': [7,3,16],
                'weight': [3,3,16,16]
            },
            'group': [2, 1],
            'stride': 1
        },
        #4
        {
            'input': [7,7,128],
            'filter': [1,1,128,16],
            'output': [7,7,16],
            'tile': {
                'input': [7,3,64],
                'weight': [1,1,64,16]
            },
            'group': [2, 1],
            'stride': 1
        },
        #5
        {
            'input': [7,7,16],
            'filter': [1,1,16,64],
            'output': [7,7,64],
            'tile': {
                'input': [7,3,16],
                'weight': [1,1,16,64]
            },
            'group': [2, 1],
            'stride': 1
        },
        #6
        {
            'input': [7,7,16],
            'filter': [3,3,16,64],
            'output': [7,7,64],
            'tile': {
                'input': [7,3,16],
                'weight': [3,3,16,16]
            },
            'group': [2, 1],
            'stride': 1
        },
        #7
        {
            'input': [7,7,128],
            'filter': [1,1,128,32],
            'output': [7,7,32],
            'tile': {
                'input': [7,3,64],
                'weight': [1,1,64,32]
            },
            'group': [2, 1],
            'stride': 1
        },
        #8
        {
            'input': [7,7,32],
            'filter': [1,1,32,128],
            'output': [7,7,128],
            'tile': {
                'input': [7,3,32],
                'weight': [1,1,32,128]
            },
            'group': [2, 1],
            'stride': 1
        },
        #9
        {
            'input': [7,7,32],
            'filter': [3,3,32,128],
            'output': [7,7,128],
            'tile': {
                'input': [7,3,32],
                'weight': [3,3,32,8]
            },
            'group': [2, 1],
            'stride': 1
        },
        #10
        {
            'input': [7,7,256],
            'filter': [1,1,256,10],
            'output': [7,7,10],
            'tile': {
                'input': [7,3,128],
                'weight': [1,1,128,10]
            },
            'group': [2, 1],
            'stride': 1
        }
    ]
}
