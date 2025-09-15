import pickle

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

features = preprocessor['features']

print(len(features), "features found:")
for feat in features:
    print(feat)
