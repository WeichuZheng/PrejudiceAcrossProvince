import torch
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
class ImageCaptionModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 图像描述模型 (轻量级)
        self.caption_processor = BlipProcessor.from_pretrained("../models/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "../models/blip-image-captioning-base"
        ).to(self.device).eval()
    
    def analyze(self, image_path):
        # 图像描述
                # 生成图像描述
        image = Image.open(image_path).convert("RGB")
        inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            caption_ids = self.caption_model.generate(**inputs, max_length=50)
        caption = self.caption_processor.decode(caption_ids[0], skip_special_tokens=True)
        return caption
    
class ImageSceneModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 语义分割模型 (轻量级)
        self.semseg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.semseg_model.to(self.device).eval()
        self.semseg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # PASCAL VOC class labels (DeepLabV3 is trained on PASCAL VOC)
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
    
    def analyze(self, image_path):
        # 语义分割提取重要元素
        img_cv = cv2.imread(image_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for proper tensor conversion
        from PIL import Image
        img_pil = Image.fromarray(img_cv)
        img_tensor = self.semseg_transform(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.semseg_model(img_tensor)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        
        # 统计主要物体占比
        unique, counts = np.unique(output_predictions, return_counts=True)
        total_pixels = output_predictions.size
        object_importance = {}
        
        for idx, count in zip(unique, counts):
            if count / total_pixels > 0.05:  # 过滤小面积物体
                if idx < len(self.classes):  # 确保索引在范围内
                    obj_name = self.classes[idx]
                    object_importance[obj_name] = count / total_pixels
        
        # 归一化重要性
        if object_importance:  # 避免除零错误
            total = sum(object_importance.values())
            normalized_importance = {k: v/total for k, v in object_importance.items()}
            return normalized_importance
        else:
            return {}

class Distiller:
    def __init__(self):
        self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def distill_concepts(self, candidates, head=None):
        self.representative_candidates = []
        """概念蒸馏：聚类去重"""
        if len(candidates) < 3:
            return candidates
        
        # 语义嵌入
        embeddings = self.semantic_model.encode(candidates)
        
        # 自适应聚类
        clustering = DBSCAN(eps=0.5, min_samples=1).fit(embeddings)
        
        # 选择每类代表词
        representative_candidates = []
        for label in set(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == label)[0]
            if len(cluster_indices) == 1:
                representative_candidates.append(candidates[cluster_indices[0]])
            else:
                # 选择最中心的词
                cluster_embeddings = embeddings[cluster_indices]
                centroid = cluster_embeddings.mean(axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                rep_idx = cluster_indices[np.argmin(distances)]
                representative_candidates.append(candidates[rep_idx])
        
        self.representative_candidates = representative_candidates
        if head is not None:
            if head > len(representative_candidates):
                head = len(representative_candidates)
            if head > 0:
                return representative_candidates[:head]  # 最多展示15个概念
        else:
            return representative_candidates
    
    def get_next_concept(self):
        # get the next concept from the list of representative candidates, remain representative_candidates list unchanged
        return next(iter(self.representative_candidates), None)

class ImageTextMatchingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = SentenceTransformer('clip-ViT-B-32', device=self.device)

    def score_attributes(self, image_path, attributes, tol=0):
        """CLIP相似度评分"""
        image = Image.open(image_path).convert("RGB")
        img_embedding = self.clip_model.encode(image, convert_to_tensor=True)
        text_embeddings = self.clip_model.encode(attributes, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(img_embedding, text_embeddings)[0]
        if tol > 0:
            return {attr: float(score) for attr, score in zip(attributes, cos_scores) if score > tol}
        else:
            return {attr: float(score) for attr, score in zip(attributes, cos_scores)}

