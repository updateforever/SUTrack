class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/wyp/project/SOITrack-main/result'  # Base directory
        self.lasot_dir = '/mnt/first/hushiyu/SOT/LaSOT/data'
        self.got10k_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data'
        self.BioDrone_dir = '/mnt/second/hushiyu/UAV/BioDrone'
        self.trackingnet_dir = '/mnt/second/wangyipei/trackingnet'
        self.coco_dir = '/mnt/second/wangyipei/coco_root'
        self.otb_dir = '/mnt/first/hushiyu/SOT/OTB/data'
        self.vot_dir = '/mnt/first/hushiyu/SOT/VOT2016/data'  # vot16S
        self.imagenet_dir = ''
        self.imagenetdet_dir = ''
        self.videocube_dir = '/mnt/first/hushiyu/SOT/VideoCube'
        #
        self.lasotSOI_dir = '/mnt/first/hushiyu/SOT/LaSOT/data'  # '/mnt/second/wangyipei/SOI/data/lasot/data'
        self.got10kSOI_dir = '/mnt/first/hushiyu/SOT/GOT-10k/data'  # '/mnt/second/wangyipei/SOI/GOT-10k/data'
        self.videocubeSOI_dir = '/mnt/first/hushiyu/SOT/VideoCube'

    def find_root_dir(self, dataset_str=None):
        if dataset_str is None:
            print('no dataset choose')
            return None
        elif dataset_str == 'lasot':
            return self.lasot_dir
        elif dataset_str == 'got10k':
            return self.got10k_dir
        elif dataset_str == 'coco':
            return self.coco_dir
        elif dataset_str == 'otb':
            return self.otb_dir
        elif dataset_str == 'lasotSOI':
            return self.lasotSOI_dir
        elif dataset_str == 'got10kSOI':
            return self.got10kSOI_dir
        elif dataset_str == 'videocube':
            return self.videocube_dir
        elif dataset_str == 'videocubeSOI':
            return self.videocubeSOI_dir
        elif dataset_str == 'vot':
            return self.vot_dir
        elif dataset_str == 'votSOI':
            return self.vot_dir

