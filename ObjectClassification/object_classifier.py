def atoi(text) : 
    return int(text) if text.isdigit() else text

def natural_keys(text) :
    return [atoi(c) for c in re.split('(\d+)', text)]

def getFilenames(path):
    root_path=""
    filenames = []
    for root, dirnames, filenames in os.walk(path):
        filenames.sort(key = natural_keys)
        rootpath = root
    return rootpath,filenames

def readImages(rootpath,filenames):
    images = []
    for filename in filenames :
        filepath = os.path.join(root,filename)
        image = ndimage.imread(filepath, mode = "L")
        images.append(image)
    return images

def chooseRandom(images):
    train_indices = np.random.choice(len(images),int(0.7*len(images)),replace = False)
    train_images = []
    test_images = []
    for i in train_indices:
        train_images.append(images[i])
    test_indices = [x for x in range(len(images)) if x not in train_indices]
    for i in test_indices:
        test_images.append(images[i])
    return train_images,test_images

def extract_features_sift(image):
    sift_object = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift_object.detectAndCompute(image, None)
    return descriptors

def kmeans(descriptor,number_of_clusters):
    feature_size = descriptor[0]
    vStack = np.array(feature_size)
    for remaining in descriptor_list[1:]:
        vStack = np.vstack((vStack, remaining))
    formatted_descriptor = vStack.copy()
    k_means_object = KMeans(n_clusters=number_of_clusters,n_jobs=-1)
    k_means_ret_obj = k_means_object.fit_predict(formatted_descriptor)
    return [k_means_object,k_means_ret_obj]

def bag_of_words(descriptor_list,k_means_ret_obj,number_of_clusters,number_of_images):
    final_histogram = np.array([np.zeros(number_of_clusters) for i in range(number_of_images)])
    count = 0
    for i in range(number_of_images):
        length = len(descriptor_list[i])
        for j in range(length):
            idx = k_means_ret_obj[count+j]
            final_histogram[i][idx] += 1
        count += length
    print ("Vocabulary Histogram Generated")
    return final_histogram


def scale_histogram(final_histogram):
    scale = StandardScaler().fit(final_histogram)
    final_histogram = scale.transform(final_histogram)
    return [scale,final_histogram]

def train(final_histogram,train_labels):
    classifier = GaussianNB()
    classifier.fit(final_histogram,train_labels)
    print("Training completed")
    return classifier

def predict_image(classfier,scale,k_means_object,test_image,number_of_clusters):
    desc = extract_features_sift(test_image)
    vocab = np.array( [[ 0 for i in range(number_of_clusters)]])
    k_means_test_ret_obj = k_means_object.predict(desc)
    for i in k_means_test_ret_obj:
            vocab[0][i] += 1
    print(vocab)
    vocab = scale.transform(vocab)
    label = classifier.predict(vocab)
    return label

