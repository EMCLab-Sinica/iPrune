config = {
    'HAR': [
        {
            'input': [1,9,1,128],
            'filter': [18,9,1,2],
            'output': [1,18,1,128],
            'tile': {
                'input': [1,9,1,64],
                'weight': [6,9,1,1],
                'output': [1,6,1,64],
            },
            'group': [6, 9],
            'pads': [0, 1, 0, 0],
            'stride': [1, 1]
        },
        {
            'input': [1,18,1,64],
            'filter': [36,18,1,2],
            'output': [1,36,1,64],
            'tile': {
                'input': [1,18,1,36],
                'weight': [6,18,1,1],
                'output': [1,6,1,36],
            },
            'group': [6, 18],
            'pads': [0, 1, 0, 0],
            'stride': [1, 1]
        },
        {
            'input': [1,36,1,32],
            'filter': [72,36,1,2],
            'output': [1,72,1,32],
            'tile': {
                'input': [1,36,1,18],
                'weight': [6,36,1,1],
                'output': [1,6,1,18],
            },
            'group': [6, 36],
            'pads': [0, 1, 0, 0],
            'stride': [1, 1]
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
    'KWS_CNN_S': [
        {
            'input': [1,1,49,10],
            'filter': [28,1,10,4],
            'output': [1,28,40,7],
            'tile': {
                'input': [1,1,49,6],
                'weight': [4,1,1,1],
                'output': [1,4,40,3],
            },
            'group': [4, 1],
            'pads': [0, 0, 0, 0],
            'stride': [1, 1]
        },
        {
            'input': [1,28,40,7],
            'filter': [28,28,10,4],
            'output': [1,28,16,4],
            'tile': {
                'input': [1,28,10,4],
                'weight': [4,28,1,1],
                'output': [1,4,1,1],
            },
            'group': [4, 28],
            'pads': [0, 0, 0, 0],
            'stride': [2, 1]
        },
        {
            'input': [1,28*16*4,1,1],
            'filter': [16,28*16*4,1,1],
            'output': [1,16,1,1],
            'tile': {
                'input': [1,64,1,1],
                'weight': [4,64,1,1],
                'output': [1,4,1,1],
            },
            'group': [4, 64],
            'stride': 1
        },
        {
            'input': [1,16,1,1],
            'filter': [128,16,1,1],
            'output': [1,128,1,1],
            'tile': {
                'input': [1,16,1,1],
                'weight': [4,16,1,1],
                'output': [1,4,1,1],
            },
            'group': [4, 16],
            'stride': 1
        },
        {
            'input': [1,128,1,1],
            'filter': [12,128,1,1],
            'output': [1,12,1,1],
            'tile': {
                'input': [1,32,1,1],
                'weight': [4,32,1,1],
                'output': [1,4,1,1],
            },
            'group': [4, 32],
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
                'input': [1,3,31,11],
                'weight': [4,3,1,1],
                'output': [1,4,15,5],
            },
            'group': [4, 3],
            'pads': [0,0,0,0],
            'stride': [2, 2]
        },
        #1
        {
            'input': [1,64,7,7],
            'filter': [16,64,1,1],
            'output': [1,16,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        },
        #2
        {
            'input': [1,16,7,7],
            'filter': [64,16,1,1],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        },
        #3
        {
            'input': [1,16,7,7],
            'filter': [64,16,3,3],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [1,1,1,1],
            'stride': [1, 1]
        },
        #4
        {
            'input': [1,128,7,7],
            'filter': [16,128,1,1],
            'output': [1,16,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        },
        #5
        {
            'input': [1,16,7,7],
            'filter': [64,16,1,1],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        },
        #6
        {
            'input': [1,16,7,7],
            'filter': [64,16,3,3],
            'output': [1,64,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [1,1,1,1],
            'stride': [1, 1]
        },
        #7
        {
            'input': [1,128,7,7],
            'filter': [32,128,1,1],
            'output': [1,32,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        },
        #8
        {
            'input': [1,32,7,7],
            'filter': [128,32,1,1],
            'output': [1,128,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        },
        #9
        {
            'input': [1,32,7,7],
            'filter': [128,32,3,3],
            'output': [1,128,7,7],
            'tile': {
                'input': [1,16,7,6],
                'weight': [4,16,1,1],
                'output': [1,4,7,6],
            },
            'group': [4, 16],
            'pads': [1,1,1,1],
            'stride': [1, 1]
        },
        #10
        {
            'input': [1,256,7,7],
            'filter': [10,256,1,1],
            'output': [1,10,7,7],
            'tile': {
                'input': [1,16,7,7],
                'weight': [2,16,1,1],
                'output': [1,2,7,7],
            },
            'group': [2, 16],
            'pads': [0,0,0,0],
            'stride': [1, 1]
        }
    ],
    'YOLOv5n': [
        #0
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
        #59
        {
            'tile': {
                'output': [0, 2, 0, 0],
            }
        },
    ]
}
