import re


class calculator():
    def __init__(self, string: str = '1+1'):
        self.string = string

    def calculate(self):
        def get_numbers(string):
            # получаем чиселки
            return [x for x in map(float, re.findall(r"(?:\d*\.\d+|\d+)", string))]

        def get_order(string):
            """
            получаем порядок операций
            :param string: исходная строка
            :return:
                symbols - все базовые математические операции и скобочки из строки
                order - порядок операций
            """
            symbols = re.findall(r"[-+*/()]", string)
            order = [x for x in symbols if x not in '()']
            counter_order = 0
            for i, s in enumerate(symbols):
                if i + 1 != len(symbols):
                    if s in '(':
                        k = i + 1
                        while symbols[k + 1] != ')':
                            if symbols[k + 1] in '*/':
                                order[counter_order] = symbols[k + 1]
                                counter_order += 1
                            elif symbols[k + 1] in '+-':
                                order[counter_order] = symbols[k + 1]
                                counter_order += 1
                            k += 1
                    elif s in '*/':
                        order[counter_order] = symbols[i + 1]
                        counter_order += 1
                    elif symbols[i + 1] in '+-':
                        order[counter_order] = symbols[i + 1]
                        counter_order += 1
                else:
                    order[counter_order] = s
            print(order)
            print(symbols)
            return order, symbols

        def __add__(current, other):
            return current + other

        def __sub__(current, other):
            return current - other

        def __mul__(current, other):
            return current * other

        def __truediv__(current, other):
            try:
                return current / other
            except ZeroDivisionError:
                return "Divide by 0 Error"

        def __neg__(current):
            # надо доработать или изменить
            return -current

        self.list = get_numbers(self.string)
        self.order, self.symbols = get_order(self.string)
        for i in range(len(self.list) - 1):
            if self.order[i] == '+':
                self.list[i + 1] = __add__(self.list[i], self.list[i + 1])
            if self.order[i] == '-':
                self.list[i + 1] = __sub__(self.list[i], self.list[i + 1])
            if self.order[i] == '*':
                self.list[i + 1] = __mul__(self.list[i], self.list[i + 1])
            if self.order[i] == '/':
                self.list[i + 1] = __truediv__(self.list[i], self.list[i + 1])
                if isinstance(self.list[i + 1], str):
                    return self.list[i + 1]
        if int(self.list[-1]) == self.list[-1]:
            return int(self.list[-1])
        return self.list[-1]


if __name__ == '__main__':
    print(calculator("4-8*2+6*(2/3.123)").calculate())
