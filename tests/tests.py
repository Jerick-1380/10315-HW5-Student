from .test_lib import Test, TestSuite
from src import pca, kmeans, boxcar, linear, rbf, polynomial
import numpy as np
import numdifftools as nd

ABSOLUTE_TOLERANCE = 1e-5

################################################### Testing PCA ###################################################

pca_tests = TestSuite("Testing PCA")
X_pca = np.array([[0.08307042, 0.32274563, 0.37534289, 0.7426287 , 0.39798024,
        0.48057114, 0.07972084, 0.8440798 , 0.97548652, 0.61562405],
       [0.96605144, 0.02892874, 0.98881999, 0.97047229, 0.10263405,
        0.19608541, 0.33336853, 0.90397422, 0.45622695, 0.42635096],
       [0.47902911, 0.23888074, 0.64802346, 0.82771305, 0.83990345,
        0.9380466 , 0.05159898, 0.6757813 , 0.87185837, 0.90282408],
       [0.58405608, 0.18509251, 0.71324132, 0.23201299, 0.63173592,
        0.615379  , 0.63850502, 0.69178487, 0.62808005, 0.71383789],
       [0.10176624, 0.60310158, 0.24958041, 0.87148584, 0.37946375,
        0.40071923, 0.78054181, 0.59484804, 0.19941424, 0.97927636],
       [0.38763738, 0.03621382, 0.37160247, 0.75691196, 0.22003911,
        0.07801999, 0.31372286, 0.61497574, 0.18804769, 0.3693253 ],
       [0.7347006 , 0.00662943, 0.25971237, 0.189447  , 0.94103705,
        0.98056122, 0.49628816, 0.44806456, 0.5117735 , 0.50438209],
       [0.10428128, 0.65017107, 0.8624467 , 0.10054595, 0.68180066,
        0.67996104, 0.97134397, 0.2544696 , 0.94027045, 0.99546687],
       [0.73237894, 0.80080807, 0.35525227, 0.07960919, 0.63934274,
        0.06622316, 0.60106312, 0.76125269, 0.29319447, 0.17312407],
       [0.09219435, 0.21599997, 0.11337929, 0.05781244, 0.82033226,
        0.86454105, 0.61906142, 0.84630161, 0.52803986, 0.26544027],
       [0.82982213, 0.13601271, 0.98875514, 0.49488068, 0.84471001,
        0.99852274, 0.25337068, 0.46980124, 0.20750848, 0.12485283],
       [0.57475443, 0.35284889, 0.63183989, 0.07548596, 0.58492005,
        0.088714  , 0.43946242, 0.22141549, 0.2182    , 0.13719802],
       [0.35088276, 0.61627665, 0.10298152, 0.20104788, 0.18865477,
        0.19748699, 0.10757399, 0.52550886, 0.44855325, 0.95785342],
       [0.7219259 , 0.50540389, 0.54467054, 0.81104893, 0.6471632 ,
        0.24850822, 0.73702514, 0.88508815, 0.72459633, 0.18121301],
       [0.33118573, 0.04944851, 0.83611257, 0.86351437, 0.03047874,
        0.49131354, 0.39142823, 0.56595443, 0.05761308, 0.71590346],
       [0.04809893, 0.92707516, 0.41246598, 0.87003247, 0.1129784 ,
        0.28563155, 0.95556001, 0.71883074, 0.09154361, 0.83184671],
       [0.8312355 , 0.07192873, 0.68667368, 0.61241054, 0.81720157,
        0.25850485, 0.72014527, 0.36430319, 0.72748171, 0.94356634],
       [0.70689868, 0.51341232, 0.68316136, 0.86923315, 0.53275868,
        0.51428514, 0.48885105, 0.68894382, 0.08985095, 0.97582983],
       [0.98542139, 0.4369878 , 0.89475575, 0.41707858, 0.16783667,
        0.99187065, 0.6757356 , 0.48973908, 0.7852244 , 0.40573622],
       [0.86021025, 0.50490852, 0.53303675, 0.62409637, 0.43559787,
        0.89937093, 0.89023843, 0.81268422, 0.96699211, 0.6561713 ],
       [0.34983709, 0.77317464, 0.43733552, 0.42417207, 0.59507112,
        0.16448762, 0.624577  , 0.41483691, 0.66288947, 0.73751154],
       [0.15021255, 0.75062801, 0.5060018 , 0.1583493 , 0.80479057,
        0.93741179, 0.89005914, 0.74699422, 0.63367491, 0.8905273 ],
       [0.40693061, 0.3471677 , 0.05553331, 0.36802344, 0.36836369,
        0.95312926, 0.2501094 , 0.36700053, 0.43622962, 0.30118637],
       [0.74918703, 0.42216553, 0.76626177, 0.34677142, 0.35955329,
        0.73305692, 0.92097452, 0.33750803, 0.07652482, 0.47767159],
       [0.36809785, 0.71418359, 0.53261572, 0.35358736, 0.60232414,
        0.53137357, 0.19634497, 0.00900154, 0.97758909, 0.59158614]])



true_pca_result_1 = np.array([[-0.07618174, -0.18423862],
       [-0.78776947,  0.66460783],
       [ 0.2625436 ,  0.22528794],
       [ 0.24115094,  0.15237509],
       [-0.39718888, -0.64688177],
       [-0.69939704,  0.09641318],
       [ 0.58199859,  0.39220193],
       [ 0.65197929, -0.50097808],
       [-0.0690656 , -0.09627636],
       [ 0.53958073, -0.11245241],
       [ 0.18441736,  0.87959507],
       [-0.08907508,  0.17516401],
       [-0.19867874, -0.53352096],
       [-0.18879533,  0.17448357],
       [-0.69218885,  0.15840594],
       [-0.61329933, -0.80185896],
       [ 0.03104047,  0.18678414],
       [-0.41950919,  0.01308047],
       [ 0.23254943,  0.50870978],
       [ 0.28556889,  0.14095678],
       [ 0.01490094, -0.48023212],
       [ 0.62031222, -0.49808304],
       [ 0.21821981,  0.06026901],
       [-0.03050313,  0.21402432],
       [ 0.39739012, -0.18783673]])

true_pca_result_2 = np.array([[-7.61817423e-02, -1.84238618e-01,  2.14646690e-01,
        -7.43220919e-01, -1.80474080e-01],
       [-7.87769472e-01,  6.64607830e-01,  2.56720160e-01,
        -6.68477552e-04, -2.48128314e-01],
       [ 2.62543597e-01,  2.25287939e-01,  5.75871518e-01,
        -5.82122679e-01, -1.63951291e-02],
       [ 2.41150943e-01,  1.52375093e-01,  9.38148405e-02,
         9.08742861e-02, -2.95512857e-02],
       [-3.97188883e-01, -6.46881766e-01,  1.78738177e-01,
         1.22693603e-02,  3.06116141e-01],
       [-6.99397039e-01,  9.64131777e-02, -3.13333683e-01,
        -2.99445925e-01,  2.68531947e-02],
       [ 5.81998589e-01,  3.92201926e-01, -2.03854480e-01,
        -1.63314423e-01,  2.52911617e-01],
       [ 6.51979294e-01, -5.00978077e-01,  3.22836168e-01,
         3.43643268e-01, -1.77800270e-01],
       [-6.90656005e-02, -9.62763612e-02, -7.22281236e-01,
         3.29220987e-01, -2.81643129e-01],
       [ 5.39580729e-01, -1.12452413e-01, -4.29637127e-01,
        -2.71752449e-01,  4.10488796e-01],
       [ 1.84417357e-01,  8.79595067e-01, -1.40306294e-01,
         4.15316063e-02,  2.66467760e-01],
       [-8.90750811e-02,  1.75164008e-01, -7.62227259e-01,
         1.91564245e-01, -2.53651576e-01],
       [-1.98678738e-01, -5.33520960e-01, -2.85150586e-01,
        -3.77720917e-01, -1.75381195e-01],
       [-1.88795329e-01,  1.74483566e-01, -7.01058026e-03,
         1.46807564e-01, -3.17518903e-01],
       [-6.92188849e-01,  1.58405939e-01,  1.93004908e-01,
        -9.91705256e-02,  3.44319563e-01],
       [-6.13299328e-01, -8.01858957e-01,  1.24357736e-01,
         3.20952660e-01,  2.97223027e-01],
       [ 3.10404658e-02,  1.86784143e-01,  2.99870333e-01,
         1.42507894e-01, -3.91250627e-01],
       [-4.19509186e-01,  1.30804715e-02,  3.06615855e-01,
         1.42211524e-01,  1.78953326e-01],
       [ 2.32549431e-01,  5.08709776e-01,  2.64501876e-01,
         3.28727857e-01, -4.27350209e-02],
       [ 2.85568894e-01,  1.40956779e-01,  4.82886861e-01,
         2.15249122e-01, -5.17323738e-02],
       [ 1.49009351e-02, -4.80232122e-01, -8.27607631e-02,
         9.63557388e-02, -3.66848308e-01],
       [ 6.20312219e-01, -4.98083042e-01,  1.86286807e-01,
         2.01816422e-01,  2.63909039e-01],
       [ 2.18219807e-01,  6.02690078e-02, -3.64817094e-01,
        -4.11506090e-01,  3.49286042e-01],
       [-3.05031311e-02,  2.14024322e-01, -1.32755282e-01,
         6.00206618e-01,  3.11132498e-01],
       [ 3.97390120e-01, -1.87836727e-01, -5.60175462e-02,
        -2.55016749e-01, -4.74550792e-01]])



def pca_test_1():
    pca_result_1 = pca(X_pca, 2)
    assert np.shape(pca_result_1) == np.shape(true_pca_result_1)
pca_tests.add_test(Test("Testing shape of PCA (dim 2)", pca_test_1))

def pca_test_2():
    pca_result_1 = pca(X_pca, 2)
    np.testing.assert_allclose(pca_result_1, true_pca_result_1, rtol=0, atol=ABSOLUTE_TOLERANCE)
pca_tests.add_test(Test("Testing values of PCA (dim 2)", pca_test_2))

def pca_test_3():
    pca_result_2 = pca(X_pca, 5)
    assert np.shape(pca_result_2) == np.shape(true_pca_result_2)
pca_tests.add_test(Test("Testing shape of PCA (dim 5)", pca_test_3))

def pca_test_4():
    pca_result_2 = pca(X_pca, 5)
    np.testing.assert_allclose(pca_result_2, true_pca_result_2, rtol=0, atol=ABSOLUTE_TOLERANCE)
pca_tests.add_test(Test("Testing values of PCA (dim 5)", pca_test_4))

################################################# Testing K-means #################################################
kmeans_tests = TestSuite("Testing K-means")

X_kmeans = np.array([[0.85745436, 0.98771417, 0.88327135, 0.44211814],
       [0.73211608, 0.94682821, 0.42469956, 0.06038805],
       [0.71739969, 0.18591835, 0.42182926, 0.2495084 ],
       [0.4139848 , 0.75422917, 0.22952695, 0.34383521],
       [0.86309863, 0.8111294 , 0.97695723, 0.6351153 ],
       [0.96780579, 0.15852174, 0.42077714, 0.547156  ],
       [0.16449633, 0.32982197, 0.76633221, 0.28543477],
       [0.58763778, 0.6755313 , 0.31472448, 0.92860224],
       [0.15154097, 0.49311517, 0.77619639, 0.3564894 ],
       [0.1310369 , 0.3528326 , 0.29548646, 0.10281993],
       [0.5848144 , 0.68645537, 0.60936571, 0.23039446],
       [0.2760856 , 0.23615207, 0.50131436, 0.40812057],
       [0.85661216, 0.54240076, 0.28681622, 0.24629453],
       [0.9140992 , 0.73987607, 0.30081886, 0.60368661],
       [0.29439769, 0.70214715, 0.32768608, 0.26424292],
       [0.60877093, 0.01767227, 0.50703855, 0.29841752],
       [0.91722534, 0.21928955, 0.1740566 , 0.44656437],
       [0.47894956, 0.27649594, 0.73814775, 0.02592561],
       [0.73384253, 0.43762842, 0.29026986, 0.26404668],
       [0.29692869, 0.44156059, 0.17858942, 0.10774039]])


true_kmeans_2_centroid = np.array([[0.80557249, 0.80356274, 0.61894298, 0.65238057],
        [0.52037546, 0.42381683, 0.43425828, 0.26483618]])

true_kmeans_2_labels = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])

true_kmeans_5_centroid = np.array([[0.85745436, 0.98771417, 0.88327135, 0.44211814],
        [0.68154407, 0.72422015, 0.36099196, 0.40220018],
        [0.78900885, 0.20380606, 0.36279428, 0.36113859],
        [0.25620511, 0.40458936, 0.51196467, 0.22153908],
        [0.86309863, 0.8111294 , 0.97695723, 0.6351153 ]])

true_kmeans_5_labels = np.array([0, 1, 2, 1, 4, 2, 3, 1, 3, 3, 1, 3, 1, 1, 3, 2, 2, 3, 2, 3])


def kmeans_test_1():
    test_kmeans_2_centroid, test_kmeans_2_labels = kmeans(X_kmeans,2)
    assert np.shape(test_kmeans_2_centroid) == np.shape(true_kmeans_2_centroid)

def kmeans_test_2():
    test_kmeans_2_centroid, test_kmeans_2_labels = kmeans(X_kmeans,2)
    assert np.shape(test_kmeans_2_labels) == np.shape(true_kmeans_2_labels)

def kmeans_test_3():
    test_kmeans_2_centroid, test_kmeans_2_labels = kmeans(X_kmeans,2)
    np.testing.assert_allclose(test_kmeans_2_centroid, true_kmeans_2_centroid, rtol=0, atol=ABSOLUTE_TOLERANCE)

def kmeans_test_4():
    test_kmeans_2_centroid, test_kmeans_2_labels = kmeans(X_kmeans,2)
    np.testing.assert_allclose(test_kmeans_2_labels, true_kmeans_2_labels, rtol=0, atol=ABSOLUTE_TOLERANCE)

def kmeans_test_5():
    test_kmeans_5_centroid, test_kmeans_5_labels = kmeans(X_kmeans,5)
    np.testing.assert_allclose(test_kmeans_5_centroid, true_kmeans_5_centroid, rtol=0, atol=ABSOLUTE_TOLERANCE)

def kmeans_test_6():
    test_kmeans_5_centroid, test_kmeans_5_labels = kmeans(X_kmeans,5)
    np.testing.assert_allclose(test_kmeans_5_labels, true_kmeans_5_labels, rtol=0, atol=ABSOLUTE_TOLERANCE)


kmeans_tests.add_test(Test("Testing shape of kmeans clusters (K = 2)", kmeans_test_1))
kmeans_tests.add_test(Test("Testing shape of kmeans labels (K = 2)", kmeans_test_2))
kmeans_tests.add_test(Test("Testing values of kmeans clusters (K = 2)", kmeans_test_3))
kmeans_tests.add_test(Test("Testing values of kmeans labels (K = 2)", kmeans_test_4))
kmeans_tests.add_test(Test("Testing values of kmeans clusters (K = 5)", kmeans_test_5))
kmeans_tests.add_test(Test("Testing values of kmeans labels (K = 5)", kmeans_test_6))


############################################ Testing Kernels ############################################

kernel_tests = TestSuite("Testing Kernels")

x = np.array([0.76579442, 0.08585739, 0.27790626, 0.52433303, 0.74076144,
       0.8340496 , 0.92175241, 0.63357006, 0.54358028, 0.1412476 ])
z = np.array([0.2476972 , 0.06244911, 0.89760641, 0.30924843, 0.63326517,
       0.12270706, 0.37657364, 0.76239715, 0.1742207 , 0.3148513 ])

true_boxcar = 1.0

true_linear = 2.147403254888077

true_rbf = 0.8439004819055

true_polynomial = 98.13175331649401

def kernel_test_1():
    test_boxcar = boxcar(x,z,3)
    np.testing.assert_allclose(test_boxcar, true_boxcar, rtol=0, atol=ABSOLUTE_TOLERANCE)

def kernel_test_2():
    test_linear = linear(x,z)
    np.testing.assert_allclose(test_linear, true_linear, rtol=0, atol=ABSOLUTE_TOLERANCE)

def kernel_test_3():
    test_rbf = rbf(x,z,0.1)
    np.testing.assert_allclose(test_rbf, true_rbf, rtol=0, atol=ABSOLUTE_TOLERANCE)

def kernel_test_4():
    test_polynomial = polynomial(x,z,4)
    np.testing.assert_allclose(test_polynomial, true_polynomial, rtol=0, atol=ABSOLUTE_TOLERANCE)


kernel_tests.add_test(Test("Testing Boxcar", kernel_test_1))
kernel_tests.add_test(Test("Testing Linear", kernel_test_2))
kernel_tests.add_test(Test("Testing RBF", kernel_test_3))
kernel_tests.add_test(Test("Testing Polynomial", kernel_test_4))




