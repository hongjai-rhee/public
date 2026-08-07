
#### 📝 cifar10_data.npz 파일 생성 방법 

**11장**에서 사용하는 이 데이터는 사이즈가 너무 크서 github 저장소에 업로드가 되지 않습니다.  

출판사의 자료실을 활용하거나

코랩에서 아래 코드를 실행하면 이 데이터를 여러분의 로컬 PC에 저장하여 사용 가능합니다. 

>from datasets import load_dataset

>data=load_dataset("uoft-cs/cifar10")   # datasets 모듈의 디렉토리와 파일 이름 지정

>x_train = np.stack([np.array(img) for img in data["train"]["img"]])  # 훈련용 이미지
>y_train = np.array(data["train"]["label"])                           # 훈련용 라벨
>x_test = np.stack([np.array(img) for img in data["test"]["img"]])    # 검증용 이미지
>y_test = np.array(data["test"]["label"])                             # 검증용 라벨

># 압축 파일로 저장
>np.savez_compressed(
    "cifar10_data.npz",
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test
>)
