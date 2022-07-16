from mmdet.apis import init_detector, inference_detector


def load_model(cfg_path, checkpoint_path, device='cuda:0', img_scale=(640, 640)):
    """
    load an mmdetection model

    args:
        cfg_path (str): path to mmrotate config file
        checkpoint_path (str): path to model checkpoint (.pth file)
        device (str): the device to load the model to
        img_scale (tuple(int, int)): input image shape
    """
    model = init_detector(cfg_path, checkpoint_path, device=device)

    # change pipeline image scale to desired scale
    test_pipeline = model.cfg.data.test.pipeline

    for elem in test_pipeline:
        if elem['type'] == 'MultiScaleFlipAug':
            elem['img_scale'] = img_scale

    return model


def inference(model, img):
    """
    run inference on model

    args:
        model (OrientedRCNN mmrotate model): the (loaded) model object
        img (either np.array or list): if an np.array - an image of shape HW or HWC. If list, a list of images, each as a np.array with a shape of HW or HWC
    """
    return inference_detector(model, img)