# import time

# def progress_show(current, total, title = 'Progress', data = '', width = 20):
#     """
#     Prints a progress bar given a current step and total.
#     Optional 'title' and 'data' values.
#     """
#     progress = '|' * int(current / total * width + 1)
#     bar  = '-' * (width - len(progress))
#     show = ''
#     if data:
#         show = f'-> {data}'
#     print(f'\r[{progress}{bar}] {title.capitalize()}: {current+1}/{total} {show}', end='\r')
#     if current + 1 == total:
#         print('\n')



class testing:
    def __init__(self) -> None:
        self.one = 'test one'
        self.two = 'test two'


t = testing()

def print_something(some_class, prop_to_print):
    x = getattr(some_class, prop_to_print)
    print(x)

print_something(t, 'two')

print(f"\N{grinning face} Done!")
