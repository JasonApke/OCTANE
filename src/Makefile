#Important, set the location of your netcdf libraries!
NCLOC=/home/japke/libs/netcdf
NCCPPLOC=/home/japke/libs/netcdf-cxx
#Set the location of your cuda library
CUDALOC=/usr/local/cuda-10.1

#Choose your gencode flags based on your gpu architecture
#Pascal
GENCODE_FLAGS=-gencode=arch=compute_60,code=sm_60
#Volta
#GENCODE_FLAGS=-gencode=arch=compute_70,code=sm_70
#Turing
#GENCODE_FLAGS=-gencode=arch=compute_75,code=sm_75
#Ampere
#GENCODE_FLAGS=-gencode=arch=compute_80,code=sm_80
#Possible upcoming include Lovelace and Hopper

#Choose your compiler, works with standard g++ and nvcc
CXX=g++
CUDACXX=nvcc

CXXFLAGS=-g -O3 -I $(NCCPPLOC)/include -I $(NCLOC)/include -I ../include #-fopenmp -DNDEBUG
#flags specific to cuda files
CUDACXXFLAGS=-g -O3 -I ../include # -I $(CUDALOC)/samples/common/inc #-lineinfo
#-lineinfo above gives lines for debugging if needed (when using cuda_memcheck)
#flags specific to cuda cooperative group files
CUDACXXCGFLAGS=--std=c++11 -rdc=true $(CUDACXXFLAGS)



CXXLIBS=-L $(NCLOC)/lib -lnetcdf -L $(NCCPPLOC)/lib -lnetcdf_c++4 -L$(CUDALOC)/lib64 -lcudart -lcudadevrt
#Removing opencv dependencies
#CXXLIBS=-L $(NCLOC)/lib -lnetcdf -L $(NCCPPLOC)/lib -lnetcdf_c++4 -L$(CVLOC)/lib -lopencv_core -lopencv_imgproc -lopencv_video -lopencv_videoio -lopencv_highgui -L$(CUDALOC)/lib64 -lcudart -lcudadevrt
ODIR=../build
#below is the list of all the scripts in octane
_OBJ=oct_variational_optical_flow.o oct_util.o oct_bicubic.o oct_zoom.o oct_gaussian.o oct_normalize_geo.o oct_fileread.o oct_filewrite.o oct_srsal_cuda.o oct_patch_match_optical_flow.o oct_binterp.o oct_navcal_cuda.o oct_polar_navcal_cuda.o oct_merc_navcal_cuda.o oct_pix2uv_cuda.o oct_optical_flow.o oct_interp.o main.o
OBJ= $(patsubst %,$(ODIR)/%,$(_OBJ))

octane: $(OBJ)
	$(CXX) -o $(ODIR)/$@ $^ $(ODIR)/jCGlink.o $(CXXFLAGS) $(CXXLIBS) 
$(ODIR)/%.o: %.cc $(DEPS)
		$(CXX) -c -o $@ $< $(CXXFLAGS)
$(ODIR)/%.o: %.cu $(DEPS)
		$(CUDACXX) -c -o $@ $< $(CUDACXXFLAGS) $(GENCODE_FLAGS)
#oct_variational_optical_flow uses cooperative groups which requires different compile flags from other cuda programs
$(ODIR)/oct_variational_optical_flow.o: oct_variational_optical_flow.cu $(DEPS)
		$(CUDACXX) -c -o $@ $(GENCODE_FLAGS) $(CUDACXXCGFLAGS) $<
		$(CUDACXX) -dlink -o $(ODIR)/jCGlink.o $(GENCODE_FLAGS) $@
.PHONY: clean

clean:
	rm -f $(ODIR)/*.o $(ODIR)/*.mod $(ODIR)/octane
