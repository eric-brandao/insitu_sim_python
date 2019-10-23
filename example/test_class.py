import insitu_cpp
# from example import add

# def test_add():
#     assert add(2, 3) == 5

my_dog = insitu_cpp.Pet('Pluto', 5)
print(my_dog.get_name())

def test_pet():
    my_dog = insitu_cpp.Pet('Pluto', 5)
    assert my_dog.get_name() == 'Pluto'
    assert my_dog.get_hunger() == 5
    my_dog.go_for_a_walk()
    assert my_dog.get_hunger() == 6
    print(my_dog.name)
