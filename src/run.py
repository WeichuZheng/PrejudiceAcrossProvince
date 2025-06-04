from analysis_model import ImageCaptionModel, ImageSceneModel, Distiller, ImageTextMatchingModel
from image_attributes import GenerativeImageAnalyzer
from json_utils import save_dict_to_jsonl, update_record_by_image_path, load_jsonl
image_path = "../docs/astronaut_rides_horse.png"
json_path = 'output.jsonl'
image_caption_model = ImageCaptionModel()
caption = image_caption_model.analyze(image_path)

image_scene_model = ImageSceneModel()
scene = image_scene_model.analyze(image_path)

candidate_attributes = GenerativeImageAnalyzer()
attributes = candidate_attributes.analyze_image(image_path)

#write to jsonl file
save_dict_to_jsonl({'image_path': image_path, 'caption': caption, 'scene': scene, 'attributes': attributes}, json_path)

# attributes = load_jsonl(json_path)[0]['attributes']
k = 10
similarity_tol = 0.2


distiller = Distiller()
all_candidates = distiller.distill_concepts(attributes['combined'])
print("All candidates: ", all_candidates)

image_text_matching_model = ImageTextMatchingModel()
all_candidates_matching = image_text_matching_model.score_attributes(image_path=image_path, attributes=all_candidates, tol=similarity_tol)

items = list(all_candidates_matching.items())
items.sort(key=lambda x:x[1],reverse=True)
top_k_attributes = [(key,value) for key, value in items]
print(dict(top_k_attributes[:k]))
# while len(top_k_attributes) < k:
#     attribute = distiller.get_next_concept()
#     if attribute:
#         new_attribute = image_text_matching_model.score_attributes(image_path, attribute, tol=similarity_tol)
#         if new_attribute:
#             print("Adding new attribute: ", new_attribute)
#             top_k_attributes.update(new_attribute)
#     else:
#         print("No more attributes to add")
#         break

# save image_path/caption/scene/attributes/top_k_attributes as json
update_record_by_image_path(json_path, image_path, {'top_k_attributes': dict(top_k_attributes[:k])})