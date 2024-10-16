cpp 컴파일 및 실행

brew install fluidsynth
g++ -std=c++11 -o create_wav create_wav.cpp -I/opt/homebrew/include -L/opt/homebrew/lib -lfluidsynth
./create_wav

데이터셋 만들기

python3 create_dataset.py

모델 만들기

python3 model.py <아무거나 입력>

모델 검증

python3 model.py
