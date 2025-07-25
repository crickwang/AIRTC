a = 10
def fun():
    nonlocal a
    a = a + 1
    print(a)
fun()
