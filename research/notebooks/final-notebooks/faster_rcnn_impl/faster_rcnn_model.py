import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_faster_rcnn_model(num_classes):
    """
    Returns a pre-trained Faster R-CNN model with a ResNet-50 backbone,
    with the classification head modified for the specified number of classes.

    Args:
        num_classes (int): The number of classes for the dataset (including background).

    Returns:
        torchvision.models.detection.FasterRCNN: The configured Faster R-CNN model.
    """
    # Load a model pre-trained on COCO
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == '__main__':
    # Example usage:
    # Adjust num_classes based on your dataset. VisDrone has 10 object classes (1-10).
    # Faster R-CNN implementations often require num_classes = num_object_classes + 1 (for background)
    num_classes = 11  # 10 VisDrone classes + 1 background
    model = get_faster_rcnn_model(num_classes)
    print(f"Faster R-CNN model loaded with {num_classes} classes.")
    # print(model) # Uncomment to see model structure 