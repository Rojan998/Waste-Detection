import random
import os
a = list(range (602))
random.shuffle(a)
direct_train = r"D:\Code\Waste-Detection\data\train\plastic"
direct_test = r"D:\Code\Waste-Detection\data\test\plastic"
direct_valid = r"D:\Code\Waste-Detection\data\valid\plastic"

train_folder = os.listdir(direct_train)
for num in a[:180]:
    file = train_folder[num]
    os.remove(os.path.join(direct_train, file))

test_folder = os.listdir(direct_test)
for num in a[100:]:
    file = test_folder[num]
    os.remove(os.path.join(direct_test, file))
    

valid_folder = os.listdir(direct_valid)
for num in a[0:100]+a[180:]:
    file = valid_folder[num]
    os.remove(os.path.join(direct_valid, file))