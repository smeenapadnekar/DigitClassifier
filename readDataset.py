def readAndSaveMetadata(filepath):
    # Forked from https://discussions.udacity.com/t/how-to-deal-with-mat-files/160657/3
    
    # Load the given MatLab file
    f = h5py.File(filepath, 'r')
    fn = filepath.split('/')[-1]
    
    # Create our empty dictionary
    metadata= {}
    metadata['height'] = []
    metadata['label'] = []
    metadata['left'] = []
    metadata['top'] = []
    metadata['width'] = []
    
    # define a function to pass to h5py's visititems() function
    def print_attrs(name, obj):
        vals = []
        if obj.shape[0] == 1:
            vals.append(obj[0][0])
        else:
            for k in range(obj.shape[0]):
                vals.append(f[obj[k][0]][0][0])
        metadata[name].append(vals)
    
    # Add information to metadata
    for item in f['/digitStruct/bbox']:
        f[item[0]].visititems(print_attrs)
    
    # Save to a pickle file
    pickle_file = fn + '.pickle'
    try:
      pickleData = open(pickle_file, 'wb')
      pickle.dump(metadata, pickleData, pickle.HIGHEST_PROTOCOL)
      pickleData.close()
    except Exception as e:
      print ('Unable to save data to', pickle_file, ':', e)
      raise



readAndSaveMetadata('data/test/digitStruct.mat')
readAndSaveMetadata('data/train/train_digitStruct.mat')
