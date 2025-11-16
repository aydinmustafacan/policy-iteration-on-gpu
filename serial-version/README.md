## How to compile 

### Create make file using cmake
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native" ..
```
### Compile
```bash
make -j
```
