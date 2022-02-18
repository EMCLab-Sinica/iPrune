config = {
    'LeNet_5': [
        {
            'input': [28,28,1],
            'filter': [5,5,1,8],
            'output': [28,28,8],
            'tile': {
                'input': [8,5,1],
                'weight': [5,5,1,8]
            },
            'group': [2, 1], # [n_filter, n_channel]
            'stride': 1
        },
        {
            'input': [14,14,8],
            'filter': [5,5,8,16],
            'output': [14,14,16],
            'tile': {
                'input': [8,5,8],
                'weight': [5,5,8,16]
            },
            'group': [2, 1],
            'stride': 1
        },
        {
            'input': [1,1,16*4*4],
            'filter': [1,1,16*4*4,256],
            'output': [1,1,256],
            'tile': {
                'input': [1,1,256],
                'weight': [1,1,256,4]
            },
            'group': [1, 16],
            'stride': 1
        },
        {
            'input': [1,1,256],
            'filter': [1,1,256,10],
            'output': [1,1,10],
            'tile': {
                'input': [1,1,256],
                'weight': [1,1,256,4]
            },
            'group': [1, 16],
            'stride': 1
        }
    ],
    'LeNet_5_p': [
        {
            'input': [28,28,1],
            'filter': [3,3,1,8],
            'output': [26,26,8],
            'tile': {
                'input': [8,3,1],
                'weight': [3,3,1,8]
            },
            'group': [1, 1, 2],
            'stride': 1
        },
        {
            'input': [26,26,8],
            'filter': [3,3,8,16],
            'output': [24,24,16],
            'tile': {
                'input': [8,3,8],
                'weight': [3,3,8,16]
            },
            'group': [1, 1, 2],
            'stride': 1
        },
        {
            'input': [1,1,2304],
            'filter': [1,1,2304,128],
            'output': [1,1,128],
            'tile': {
                'input': [1,1,2172],
                'weight': [1,1,2172,4]
            },
            'group': [1, 1, 2],
            'stride': 1
        },
        {
            'input': [1,1,128],
            'filter': [1,1,128,10],
            'output': [1,1,10],
            'tile': {
                'input': [1,1,128],
                'weight': [1,1,128,4]
            },
            'group': [1, 1, 2],
            'stride': 1
        }
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
