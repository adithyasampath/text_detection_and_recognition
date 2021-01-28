CXXFLAGS = -I include  -std=c++11 -O3 $(shell python-config --cflags)
LDFLAGS = $(shell python-config --ldflags)

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)


cl adaptor.cpp .\include\clipper\clipper.cpp /I .\include /I "C:\Users\Dell\anaconda3\include" /LD /Fe:adaptor.pyd /link/LIBPATH:"C:\Users\Dell\anaconda3\libs"