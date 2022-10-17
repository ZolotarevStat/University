import re


class calculator():
    def __init__(self, string='1+1'):
        """
        Инициализация всего на свете
        :param string: входная строка
        """
        self.string = string
        if self.string[0] in '+-':
            self.string = '0' + self.string
        self.operators = {'+': (1, lambda x, y: x + y), '-': (1, lambda x, y: x - y),
                          '*': (2, lambda x, y: x * y), '/': (2, lambda x, y: x / y)}
    def parse_and_calculate(self):
        """
            раскрываем исходную строку, переводим всё в польскую запись и считаем результат
            :return: ответ
        """
        self.parsed_list = re.findall(r"(?:\d*\.\d+|\d+|[-+*/()])", self.string)
        def get_polish():
            stack = []
            for token in self.parsed_list:
                if token in self.operators:
                    while stack and stack[-1] != "(" and self.operators[token][0] <= self.operators[stack[-1]][0]:
                        yield stack.pop()
                    stack.append(token)
                elif token == ")":
                    while stack:
                        x = stack.pop()
                        if x == "(":
                            break
                        yield x
                elif token == "(":
                    stack.append(token)
                else:
                    yield token
            while stack:
                yield stack.pop()

        stack = []
        for token in get_polish():
            if token in self.operators:
                y, x = float(stack.pop()), float(stack.pop())
                if token == '/':
                    try:
                        stack.append(self.operators[token][1](x, y))
                    except ZeroDivisionError:
                        return "Divide by 0 Error"
                else:
                    stack.append(self.operators[token][1](x, y))
            else:
                stack.append(token)
        if int(stack[0]) == stack[0]:
            return int(stack[0])
        return stack[0]

if __name__=='__main__':
    #calculate = calculator("-5")
    #calculate = calculator("100*2*3")
    calculate = calculator("4-8*2+6*(2/3.123)")
    print(calculate.get_generator())

