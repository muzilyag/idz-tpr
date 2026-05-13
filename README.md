```bash
git clone https://github.com/muzilyag/idz-tpr.git
```

Опционально (venv, чтобы потом удалить нахрен это)
```bash
   python -m venv venv
   source venv/bin/activate    # Linux
   venv\Scripts\activate       # Windows
```

Установка нужных модулей
```bash
pip install -r requirements.txt
```

Запуск
```bash
python main.py
```
## Формат ввода
```
Пример задачи, который можно ввести в консоли (это вариант из лр1):
F = 1x1 + 4x2 + 1x3 -> max
-1x1 + 2x2 + 1x3 = 4
3x1 + 1x2 + 2x3 <= 9
2x1 + 3x2 + 1x3 >= 6
x1 >= 0
x2 >= 0
x3 >= 0
solve
```
## Требования
- Python 3.8 или выше.