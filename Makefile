#######################################################################################################

# Mac OS X
#INCLUDE_PATH      = -I/usr/local/include/ -I/usr/local/include/eigen3/
#LIBRARY_PATH      = -L/usr/local/lib/
#OPENGL_LIBS       = -framework OpenGL -framework GLUT

# # Linux
INCLUDE_PATH      =
LIBRARY_PATH      =
OPENGL_LIBS       = -lglut -lGL -lX11

# # Windows / Cygwin
#INCLUDE_PATH      = -I/usr/include/opengl
#LIBRARY_PATH      = -L/usr/lib/w32api
#OPENGL_LIBS       = -lglut -lopengl32 -lm

#######################################################################################################

TARGET = sph
SEC_TARGET = sph-rt
CC = g++ -fopenmp
LD = g++ -fopenmp
CFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -pedantic -Wno-vla-extension $(INCLUDE_PATH) -I./include -I./src -DNDEBUG
LFLAGS = -std=c++11 -O3 -Wall -Wno-deprecated -Werror -pedantic $(LIBRARY_PATH) -DNDEBUG
LIBS = $(OPENGL_LIBS)

OBJS = obj/main.o
SEC_OBJS = obj/main-rt.o

default: $(SEC_TARGET)

all: clean $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $(LFLAGS) $(OBJS) $(LIBS) -o $(TARGET)

$(SEC_TARGET): $(SEC_OBJS)
	$(LD) $(LFLAGS) $(SEC_OBJS) $(LIBS) -o $(SEC_TARGET)

obj/main.o: src/main.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) -c src/main.cpp -o obj/main.o

obj/main-rt.o: src/main-rt.cpp
	mkdir -p obj
	$(CC) $(CFLAGS) -c src/main-rt.cpp -o obj/main-rt.o

clean:
	rm -f $(OBJS)
	rm -f $(TARGET)
	rm -f $(TARGET).exe
	rm -f $(SEC_OBJS)
	rm -f $(SEC_TARGET)
