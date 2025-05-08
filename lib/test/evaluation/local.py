from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/first/hushiyu/SOT/GOT-10k/data/got10k_lmdb'
    settings.got10k_path = '/mnt/first/hushiyu/SOT/GOT-10k/data'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/wyp/project/SOITrack-main/data/itb'
    settings.lasot_extension_subset_path_path = '/home/wyp/project/SOITrack-main/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/wyp/project/SOITrack-main/data/lasot_lmdb'
    settings.lasot_path = '/mnt/first/hushiyu/SOT/LaSOT/data'
    settings.network_path = '/home/wyp/project/SOITrack-main/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/wyp/project/SOITrack-main/data/nfs'
    settings.otb_path = '/mnt/first/hushiyu/SOT/OTB/data'


    settings.otblang_path = '/home/cx/cx1/github-repo/SUTrack/data/otb_lang'
    settings.prj_dir = '/home/cx/cx1/github-repo/SUTrack'
    settings.result_plot_path = '/home/cx/cx1/github-repo/SUTrack/test/result_plots'
    settings.results_path = '/home/cx/cx1/github-repo/SUTrack/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/cx/cx1/github-repo/SUTrack'
    settings.segmentation_path = '/home/cx/cx1/github-repo/SUTrack/test/segmentation_results'
    settings.tc128_path = '/home/cx/cx1/github-repo/SUTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/cx/cx1/github-repo/SUTrack/data/tnl2k/test'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/cx/cx1/github-repo/SUTrack/data/trackingnet'
    settings.uav_path = '/home/cx/cx1/github-repo/SUTrack/data/UAV123'
    settings.vot_path = '/home/cx/cx1/github-repo/SUTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    settings.prj_dir = '/home/wyp/project/SUTrack'
    settings.result_plot_path = '/home/wyp/project/SUTrack/output/test/result_plots'
    settings.results_path = '/home/wyp/project/SUTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/wyp/project/SUTrack/output'
    settings.soi_path = '/home/wyp/project/SUTrack/output/test/soi_results'
    settings.sot_save_dir = '/home/wyp/project/SUTrack/output/test/soi_tracking_results'
    settings.soi_online_save_dir = '/home/wyp/project/SUTrack/output/test/soi_online_tracking_results'
    settings.soi_vis_dir = '/home/wyp/project/SUTrack/output/test/soi_vis'
    return settings

