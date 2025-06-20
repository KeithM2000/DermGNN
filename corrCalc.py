import pandas as pd
def correlation_calc(image_data):
    correlations = []
    print('correlations')
    print()
    print('angular second moment')
    print(image_data['angularMoment'].corr(image_data['malignant']))
    print('contrast')
    print(image_data['contrast'].corr(image_data['malignant']))
    print('correlation')
    print(image_data['correlation'].corr(image_data['malignant']))
    print('sum of squares')
    print(image_data['soS_Var'].corr(image_data['malignant']))
    print('Inverse Difference Moment')
    print(image_data['invDifferenceMoment'].corr(image_data['malignant']))
    print('Sum Average')
    print(image_data['sumAverage'].corr(image_data['malignant']))
    print('Sum Variance')
    print(image_data['sumVariance'].corr(image_data['malignant']))
    print('Sum Entropy')
    print(image_data['sumEntropy'].corr(image_data['malignant']))
    print('entropy')
    print(image_data['entropy'].corr(image_data['malignant']))
    print('Difference Variance')
    print(image_data['differenceVariance'].corr(image_data['malignant']))
    print('Difference Entropy')
    print(image_data['differenceEntropy'].corr(image_data['malignant']))
    print('f12')
    print(image_data['imc1'].corr(image_data['malignant']))
    print('f13')
    print(image_data['imc2'].corr(image_data['malignant']))
    print('trace')
    print(image_data['trace'].corr(image_data['malignant']))