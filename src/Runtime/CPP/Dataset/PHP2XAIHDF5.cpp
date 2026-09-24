#include "PHP2XAIHDF5.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>


PHP2XAIHDF5::PHP2XAIHDF5()
{
    file_ = -1;
}


PHP2XAIHDF5::~PHP2XAIHDF5()
{
    if (file_ >= 0) {
        H5Fclose(file_);
    }
}


PHP2XAIHDF5* PHP2XAIHDF5::create(
    const std::string& filename
)
{
    hid_t file = H5Fcreate(
        filename.c_str(),
        H5F_ACC_TRUNC,
        H5P_DEFAULT,
        H5P_DEFAULT
    );

    if (file < 0) {
        throw std::runtime_error(
            "Unable to create HDF5 file: " + filename
        );
    }


    PHP2XAIHDF5* dataset =
        new PHP2XAIHDF5();


    dataset->file_ = file;


    return dataset;
}


PHP2XAIHDF5* PHP2XAIHDF5::open(
    const std::string& filename
)
{
    hid_t file = H5Fopen(
        filename.c_str(),
        H5F_ACC_RDWR,
        H5P_DEFAULT
    );

    if (file < 0) {
        throw std::runtime_error(
            "Unable to open HDF5 file: " + filename
        );
    }


    PHP2XAIHDF5* dataset =
        new PHP2XAIHDF5();


    dataset->file_ = file;


    return dataset;
}


hid_t PHP2XAIHDF5::getHDF5Type(
    DType dtype
) const
{
    switch (dtype) {

        case FLOAT32:
            return H5T_NATIVE_FLOAT;

        case FLOAT64:
            return H5T_NATIVE_DOUBLE;

        case INT32:
            return H5T_NATIVE_INT32;

        case INT64:
            return H5T_NATIVE_INT64;

        default:
            throw std::runtime_error(
                "Unsupported dtype"
            );
    }
}


PHP2XAIHDF5::DType
PHP2XAIHDF5::getDType(
    hid_t datatype
) const
{
    H5T_class_t typeClass =
        H5Tget_class(datatype);

    size_t size =
        H5Tget_size(datatype);


    if (typeClass == H5T_FLOAT) {

        if (size == 4) {
            return FLOAT32;
        }

        if (size == 8) {
            return FLOAT64;
        }
    }


    if (typeClass == H5T_INTEGER) {

        if (size == 4) {
            return INT32;
        }

        if (size == 8) {
            return INT64;
        }
    }


    throw std::runtime_error(
        "Unsupported HDF5 datatype"
    );
}


size_t PHP2XAIHDF5::getDTypeSize(
    DType dtype
) const
{
    switch (dtype) {

        case FLOAT32:
        case INT32:
            return 4;

        case FLOAT64:
        case INT64:
            return 8;

        default:
            throw std::runtime_error(
                "Unsupported dtype"
            );
    }
}


PHP2XAIHDF5::FieldInfo
PHP2XAIHDF5::openField(
    const std::string& field
) const
{
    FieldInfo info;


    /*
     * Apre il dataset HDF5.
     */
    info.dataset = H5Dopen2(
        file_,
        field.c_str(),
        H5P_DEFAULT
    );

    if (info.dataset < 0) {
        throw std::runtime_error(
            "Unable to open field: " + field
        );
    }


    /*
     * Recupera il datatype.
     */
    info.datatype =
        H5Dget_type(info.dataset);

    if (info.datatype < 0) {

        H5Dclose(info.dataset);

        throw std::runtime_error(
            "Unable to get datatype for field: " + field
        );
    }


    /*
     * Recupera il dataspace.
     */
    info.dataspace =
        H5Dget_space(info.dataset);

    if (info.dataspace < 0) {

        H5Tclose(info.datatype);
        H5Dclose(info.dataset);

        throw std::runtime_error(
            "Unable to get dataspace for field: " + field
        );
    }


    /*
     * Recupera il numero di dimensioni.
     */
    info.rank =
        H5Sget_simple_extent_ndims(
            info.dataspace
        );

    if (info.rank < 1) {

        H5Sclose(info.dataspace);
        H5Tclose(info.datatype);
        H5Dclose(info.dataset);

        throw std::runtime_error(
            "Invalid field rank: " + field
        );
    }


    /*
     * Recupera le dimensioni correnti.
     */
    info.dims.resize(info.rank);


    if (
        H5Sget_simple_extent_dims(
            info.dataspace,
            info.dims.data(),
            nullptr
        ) < 0
    ) {

        H5Sclose(info.dataspace);
        H5Tclose(info.datatype);
        H5Dclose(info.dataset);

        throw std::runtime_error(
            "Unable to get dimensions for field: " + field
        );
    }


    return info;
}


void PHP2XAIHDF5::closeField(
    FieldInfo& field
) const
{
    H5Sclose(field.dataspace);
    H5Tclose(field.datatype);
    H5Dclose(field.dataset);
}


void PHP2XAIHDF5::setField(
    const std::string& name,
    DType dtype,
    const std::vector<int64_t>& shape
)
{
    if (shape.empty()) {
        throw std::runtime_error(
            "Field shape cannot be empty"
        );
    }


    /*
     * Aggiungiamo il primo asse N.
     *
     * shape = [128]
     *
     * diventa:
     *
     * [N, 128]
     */
    int rank =
        static_cast<int>(shape.size()) + 1;


    std::vector<hsize_t> dims(rank);
    std::vector<hsize_t> maxDims(rank);
    std::vector<hsize_t> chunkDims(rank);


    /*
     * Il dataset parte con zero sample.
     */
    dims[0] = 0;

    /*
     * Il numero di sample può crescere.
     */
    maxDims[0] = H5S_UNLIMITED;


    size_t sampleElements = 1;


    for (size_t i = 0; i < shape.size(); i++) {

        if (shape[i] <= 0) {
            throw std::runtime_error(
                "Field dimensions must be greater than zero"
            );
        }


        hsize_t dimension =
            static_cast<hsize_t>(shape[i]);


        dims[i + 1] = dimension;

        maxDims[i + 1] = dimension;

        chunkDims[i + 1] = dimension;


        sampleElements *=
            static_cast<size_t>(shape[i]);
    }


    /*
     * Dimensione in byte di un sample.
     */
    size_t sampleBytes =
        sampleElements *
        getDTypeSize(dtype);


    /*
     * Cerchiamo chunk di circa 1 MiB.
     */
    const size_t targetChunkBytes =
        1024 * 1024;


    size_t samplesPerChunk =
        targetChunkBytes / sampleBytes;


    if (samplesPerChunk < 1) {
        samplesPerChunk = 1;
    }


    chunkDims[0] =
        static_cast<hsize_t>(
            samplesPerChunk
        );


    /*
     * Crea il dataspace.
     *
     * dims:
     *     [0, ...shape]
     *
     * maxDims:
     *     [UNLIMITED, ...shape]
     */
    hid_t dataspace =
        H5Screate_simple(
            rank,
            dims.data(),
            maxDims.data()
        );


    if (dataspace < 0) {
        throw std::runtime_error(
            "Unable to create HDF5 dataspace"
        );
    }


    /*
     * Proprietà di creazione del dataset.
     */
    hid_t properties =
        H5Pcreate(
            H5P_DATASET_CREATE
        );


    if (properties < 0) {

        H5Sclose(dataspace);

        throw std::runtime_error(
            "Unable to create HDF5 dataset properties"
        );
    }


    /*
     * Imposta il chunking.
     */
    if (
        H5Pset_chunk(
            properties,
            rank,
            chunkDims.data()
        ) < 0
    ) {

        H5Pclose(properties);
        H5Sclose(dataspace);

        throw std::runtime_error(
            "Unable to configure HDF5 chunking"
        );
    }


    /*
     * Crea il field.
     */
    hid_t dataset =
        H5Dcreate2(
            file_,
            name.c_str(),
            getHDF5Type(dtype),
            dataspace,
            H5P_DEFAULT,
            properties,
            H5P_DEFAULT
        );


    H5Pclose(properties);
    H5Sclose(dataspace);


    if (dataset < 0) {
        throw std::runtime_error(
            "Unable to create field: " + name
        );
    }


    H5Dclose(dataset);
}


PHP2XAIHDF5::FieldMetadata
PHP2XAIHDF5::fieldMetadata(
    const std::string& name
) const
{
    FieldInfo field =
        openField(name);


    FieldMetadata metadata;


    try {

        metadata.dtype =
            getDType(field.datatype);


        /*
         * dims[0] è N.
         *
         * Non fa parte della shape
         * del singolo sample.
         */
        for (int i = 1; i < field.rank; i++) {

            metadata.shape.push_back(
                static_cast<int64_t>(
                    field.dims[i]
                )
            );
        }

    }
    catch (...) {

        closeField(field);

        throw;
    }


    closeField(field);


    return metadata;
}


int64_t PHP2XAIHDF5::count() const
{
    H5G_info_t groupInfo;


    if (
        H5Gget_info(
            file_,
            &groupInfo
        ) < 0
    ) {
        throw std::runtime_error(
            "Unable to read HDF5 root group"
        );
    }


    /*
     * Nessun field.
     */
    if (groupInfo.nlinks == 0) {
        return 0;
    }


    int64_t result = -1;


    for (
        hsize_t i = 0;
        i < groupInfo.nlinks;
        i++
    ) {

        /*
         * Recuperiamo prima la lunghezza
         * del nome del field.
         */
        ssize_t nameLength =
            H5Lget_name_by_idx(
                file_,
                ".",
                H5_INDEX_NAME,
                H5_ITER_INC,
                i,
                nullptr,
                0,
                H5P_DEFAULT
            );


        if (nameLength < 0) {
            throw std::runtime_error(
                "Unable to read field name"
            );
        }


        std::vector<char> name(
            static_cast<size_t>(
                nameLength
            ) + 1
        );


        /*
         * Recuperiamo il nome.
         */
        if (
            H5Lget_name_by_idx(
                file_,
                ".",
                H5_INDEX_NAME,
                H5_ITER_INC,
                i,
                name.data(),
                name.size(),
                H5P_DEFAULT
            ) < 0
        ) {
            throw std::runtime_error(
                "Unable to read field name"
            );
        }


        /*
         * openField fa tutti i controlli
         * comuni.
         */
        FieldInfo field =
            openField(name.data());


        int64_t fieldCount =
            static_cast<int64_t>(
                field.dims[0]
            );


        closeField(field);


        /*
         * Il primo field stabilisce N.
         */
        if (result < 0) {

            result = fieldCount;

        }
        else if (result != fieldCount) {

            throw std::runtime_error(
                "Dataset fields have inconsistent sample counts"
            );
        }
    }


    return result < 0 ? 0 : result;
}


void PHP2XAIHDF5::add(
    const std::string& fieldName,
    const void* data
)
{
    if (data == nullptr) {
        throw std::runtime_error(
            "Data cannot be null"
        );
    }


    FieldInfo field =
        openField(fieldName);


    /*
     * Numero attuale di sample.
     */
    hsize_t oldCount =
        field.dims[0];


    /*
     * Estendiamo N di uno.
     */
    field.dims[0] =
        oldCount + 1;


    if (
        H5Dset_extent(
            field.dataset,
            field.dims.data()
        ) < 0
    ) {

        closeField(field);

        throw std::runtime_error(
            "Unable to extend field: " +
            fieldName
        );
    }


    /*
     * Il dataspace contenuto in FieldInfo
     * rappresentava la dimensione precedente.
     *
     * Dopo H5Dset_extent lo recuperiamo
     * nuovamente.
     */
    H5Sclose(field.dataspace);


    field.dataspace =
        H5Dget_space(
            field.dataset
        );


    if (field.dataspace < 0) {

        H5Tclose(field.datatype);
        H5Dclose(field.dataset);

        throw std::runtime_error(
            "Unable to get updated dataspace: " +
            fieldName
        );
    }


    /*
     * Selezioniamo il nuovo sample.
     *
     * start:
     *
     * [oldCount, 0, 0, ...]
     *
     * selection:
     *
     * [1, shape...]
     */
    std::vector<hsize_t> start(
        field.rank,
        0
    );

    std::vector<hsize_t> selection(
        field.rank
    );


    start[0] = oldCount;

    selection[0] = 1;


    for (int i = 1; i < field.rank; i++) {
        selection[i] = field.dims[i];
    }


    if (
        H5Sselect_hyperslab(
            field.dataspace,
            H5S_SELECT_SET,
            start.data(),
            nullptr,
            selection.data(),
            nullptr
        ) < 0
    ) {

        closeField(field);

        throw std::runtime_error(
            "Unable to select field position: " +
            fieldName
        );
    }


    /*
     * Il buffer data contiene un solo sample.
     *
     * Il relativo memory dataspace è quindi:
     *
     * [1, shape...]
     */
    hid_t memorySpace =
        H5Screate_simple(
            field.rank,
            selection.data(),
            nullptr
        );


    if (memorySpace < 0) {

        closeField(field);

        throw std::runtime_error(
            "Unable to create memory dataspace"
        );
    }


    /*
     * Scrive il sample.
     */
    if (
        H5Dwrite(
            field.dataset,
            field.datatype,
            memorySpace,
            field.dataspace,
            H5P_DEFAULT,
            data
        ) < 0
    ) {

        H5Sclose(memorySpace);
        closeField(field);

        throw std::runtime_error(
            "Unable to write field: " +
            fieldName
        );
    }


    H5Sclose(memorySpace);

    closeField(field);
}


void PHP2XAIHDF5::readIndices(
    const std::string& fieldName,
    const std::vector<int64_t>& indices,
    void* output
) const
{
    if (output == nullptr) {
        throw std::runtime_error(
            "Output cannot be null"
        );
    }


    if (indices.empty()) {
        return;
    }


    FieldInfo field = openField(fieldName);
    hid_t memorySpace = -1;

    try {
        const DType dtype = getDType(field.datatype);

        size_t sampleElements = 1;
        for (int i = 1; i < field.rank; i++) {
            const size_t dimension = static_cast<size_t>(field.dims[i]);
            if (dimension == 0 || sampleElements > std::numeric_limits<size_t>::max() / dimension)
                throw std::runtime_error("Invalid or oversized sample shape: " + fieldName);
            sampleElements *= dimension;
        }

        const size_t dtypeSize = getDTypeSize(dtype);
        if (sampleElements > std::numeric_limits<size_t>::max() / dtypeSize)
            throw std::runtime_error("Sample buffer is too large: " + fieldName);
        const size_t sampleBytes = sampleElements * dtypeSize;

        std::vector<int64_t> sortedIndices = indices;
        for (const int64_t index : sortedIndices) {
            if (index < 0 || static_cast<hsize_t>(index) >= field.dims[0])
                throw std::runtime_error("Dataset index out of range");
        }
        std::sort(sortedIndices.begin(), sortedIndices.end());
        sortedIndices.erase(
            std::unique(sortedIndices.begin(), sortedIndices.end()),
            sortedIndices.end());

        if (sortedIndices.size() > std::numeric_limits<size_t>::max() / sampleBytes)
            throw std::runtime_error("Output buffer is too large: " + fieldName);

        /* Build one file selection containing all requested samples. */
        if (H5Sselect_none(field.dataspace) < 0)
            throw std::runtime_error("Unable to clear dataset selection: " + fieldName);

        std::vector<hsize_t> start(field.rank, 0);
        std::vector<hsize_t> sampleDims = field.dims;
        sampleDims[0] = 1;

        for (size_t i = 0; i < sortedIndices.size(); ++i) {
            start[0] = static_cast<hsize_t>(sortedIndices[i]);
            const H5S_seloper_t operation = i == 0 ? H5S_SELECT_SET : H5S_SELECT_OR;

            if (H5Sselect_hyperslab(
                    field.dataspace,
                    operation,
                    start.data(),
                    nullptr,
                    sampleDims.data(),
                    nullptr) < 0) {
                throw std::runtime_error("Unable to select dataset samples: " + fieldName);
            }
        }

        /* Memory layout is [number of unique indices, sample shape...]. */
        std::vector<hsize_t> memoryDims = field.dims;
        memoryDims[0] = static_cast<hsize_t>(sortedIndices.size());
        memorySpace = H5Screate_simple(field.rank, memoryDims.data(), nullptr);
        if (memorySpace < 0)
            throw std::runtime_error("Unable to create batch memory dataspace");

        const hssize_t filePoints = H5Sget_select_npoints(field.dataspace);
        const hssize_t memoryPoints = H5Sget_simple_extent_npoints(memorySpace);
        if (filePoints < 0 || filePoints != memoryPoints)
            throw std::runtime_error("HDF5 file and memory selections have different sizes");

        std::vector<char> sortedOutput(sortedIndices.size() * sampleBytes);
        if (H5Dread(
                field.dataset,
                field.datatype,
                memorySpace,
                field.dataspace,
                H5P_DEFAULT,
                sortedOutput.data()) < 0) {
            throw std::runtime_error("Unable to read field: " + fieldName);
        }

        /* Restore the caller's requested order, including repeated indices. */
        char* outputBytes = static_cast<char*>(output);
        for (size_t i = 0; i < indices.size(); ++i) {
            const auto found = std::lower_bound(sortedIndices.begin(), sortedIndices.end(), indices[i]);
            const size_t sortedPosition = static_cast<size_t>(found - sortedIndices.begin());
            std::memcpy(
                outputBytes + i * sampleBytes,
                sortedOutput.data() + sortedPosition * sampleBytes,
                sampleBytes);
        }

        H5Sclose(memorySpace);
        closeField(field);
    }
    catch (...) {
        if (memorySpace >= 0)
            H5Sclose(memorySpace);
        closeField(field);
        throw;
    }
}
