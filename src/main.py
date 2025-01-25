from utils import transform_images, TripletImageLoader


image_tensors_train = transform_images(dataset_path="./dataset/face/train")
image_tensors_test = transform_images(dataset_path="./dataset/face/test")

triplet_image_loader = TripletImageLoader(image_tensors_train, image_tensors_test)
triplets = triplet_image_loader.get_batch(n_pick=4)
print(type(triplets))
print(triplets)