#pragma once

#include <hdf5.h>

#include <string>
#include <vector>
#include <cstdint>


class PHP2XAIHDF5
{
public:

    enum DType {
        FLOAT32 = 1,
        FLOAT64 = 2,
        INT32   = 3,
        INT64   = 4
    };


    struct FieldMetadata {
        DType dtype;
        std::vector<int64_t> shape;
    };


    static PHP2XAIHDF5* create(
        const std::string& filename
    );

    static PHP2XAIHDF5* open(
        const std::string& filename
    );


    void setField(
        const std::string& name,
        DType dtype,
        const std::vector<int64_t>& shape
    );


    FieldMetadata fieldMetadata(
        const std::string& name
    ) const;


    int64_t count() const;


    void add(
        const std::string& field,
        const void* data
    );


    void readIndices(
        const std::string& field,
        const std::vector<int64_t>& indices,
        void* output
    ) const;


    ~PHP2XAIHDF5();


private:

    struct FieldInfo {
        hid_t dataset;
        hid_t datatype;
        hid_t dataspace;

        int rank;

        std::vector<hsize_t> dims;
    };


    PHP2XAIHDF5();


    hid_t file_;


    FieldInfo openField(
        const std::string& field
    ) const;


    void closeField(
        FieldInfo& field
    ) const;


    hid_t getHDF5Type(
        DType dtype
    ) const;


    DType getDType(
        hid_t datatype
    ) const;


    size_t getDTypeSize(
        DType dtype
    ) const;
}; 
