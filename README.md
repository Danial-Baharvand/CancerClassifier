# CancerClassifier
This model builds four different types of classifiers
and evaluate their performance on the medical_records dataset which includes cancer patient records.
The classification task is to predict whether a tumour is malignant (M) or benign (B).
### Following classifiers are implemented using sklearn
• a nearest neighbours classifier

• a decision tree classifier

• a support vector machine classifier

• a neural network classifier
## Dataset
The records are stored in a text file named “medical_records.data”. Each row
corresponds to a patient record. The diagnosis is the attribute predicted. In
this dataset, the diagnosis is the second field and is either B (benign) or M
(malignant). There are 32 attributes in total (ID, diagnosis, and 30 real-valued
input features).

### Attribute Information

1. ID number

2. Diagnosis (M = malignant, B = benign)

3-32. Ten real-valued features are computed for each cell nucleus:

a) radius (mean of distances from center to points on the perimeter)

b) texture (standard deviation of gray-scale values)

c) perimeter

d) area

e) smoothness (local variation in radius lengths)

f) compactness (perimeter^2 / area - 1.0)

g) concavity (severity of concave portions of the contour)

h) concave points (number of concave portions of the contour)

i) symmetry

j) fractal dimension ("coastline approximation" - 1)
