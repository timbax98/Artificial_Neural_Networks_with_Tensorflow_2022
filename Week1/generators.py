def meow_generator(generation):
    for i in range(0,generation):
        x = 2**i
        yield(x*"Meow ")

my_meows = meow_generator(3)

for i in my_meows:
    print(i)