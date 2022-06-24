
Create build folder:
```sh
$ mkdir bin
$ cd bin
```

Configure without muhlib:
```sh
$ cmake ..
```
With muhlib:
```sh
$ cmake .. -DUSE_MUHLIB=ON
```

Compile & run:
```sh
$ make
$ ./App
```

