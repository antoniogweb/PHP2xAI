#include "ffi.hpp"

#include "PHP2XAIHDF5.hpp"

#include <exception>
#include <string>
#include <vector>

struct PHP2XAIHDF5_Handle {
	PHP2XAIHDF5* dataset = nullptr;
	std::string error;
};

namespace {
thread_local std::string globalError;

void clearError(PHP2XAIHDF5_Handle* handle)
{
	globalError.clear();
	if (handle)
		handle->error.clear();
}

void setError(PHP2XAIHDF5_Handle* handle, const char* message)
{
	globalError = message ? message : "Unknown HDF5 error";
	if (handle)
		handle->error = globalError;
}

void setCurrentException(PHP2XAIHDF5_Handle* handle)
{
	try {
		throw;
	} catch (const std::exception& e) {
		setError(handle, e.what());
	} catch (...) {
		setError(handle, "Unknown C++ exception");
	}
}

bool validHandle(PHP2XAIHDF5_Handle* handle)
{
	if (handle && handle->dataset)
		return true;
	setError(handle, "Invalid HDF5 handle");
	return false;
}
}

extern "C" {

PHP2XAIHDF5_Handle* php2xai_hdf5_create(const char* filename)
{
	globalError.clear();
	if (!filename) {
		setError(nullptr, "Filename cannot be null");
		return nullptr;
	}
	try {
		auto* handle = new PHP2XAIHDF5_Handle();
		try {
			handle->dataset = PHP2XAIHDF5::create(filename);
		} catch (...) {
			delete handle;
			throw;
		}
		return handle;
	} catch (...) {
		setCurrentException(nullptr);
		return nullptr;
	}
}

PHP2XAIHDF5_Handle* php2xai_hdf5_open(const char* filename)
{
	globalError.clear();
	if (!filename) {
		setError(nullptr, "Filename cannot be null");
		return nullptr;
	}
	try {
		auto* handle = new PHP2XAIHDF5_Handle();
		try {
			handle->dataset = PHP2XAIHDF5::open(filename);
		} catch (...) {
			delete handle;
			throw;
		}
		return handle;
	} catch (...) {
		setCurrentException(nullptr);
		return nullptr;
	}
}

void php2xai_hdf5_destroy(PHP2XAIHDF5_Handle* handle)
{
	if (!handle)
		return;
	delete handle->dataset;
	handle->dataset = nullptr;
	delete handle;
}

int php2xai_hdf5_set_field(PHP2XAIHDF5_Handle* handle, const char* name,
	int dtype, const int64_t* shape, size_t rank)
{
	clearError(handle);
	if (!validHandle(handle) || !name || !shape || rank == 0) {
		setError(handle, "Invalid set_field arguments");
		return 1;
	}
	try {
		handle->dataset->setField(name, static_cast<PHP2XAIHDF5::DType>(dtype),
			std::vector<int64_t>(shape, shape + rank));
		return 0;
	} catch (...) {
		setCurrentException(handle);
		return 2;
	}
}

int php2xai_hdf5_field_metadata(PHP2XAIHDF5_Handle* handle, const char* name,
	int* dtype, int64_t* shape, size_t shape_capacity, size_t* shape_rank)
{
	clearError(handle);
	if (!validHandle(handle) || !name || !dtype || !shape_rank) {
		setError(handle, "Invalid field_metadata arguments");
		return 1;
	}
	try {
		const auto metadata = handle->dataset->fieldMetadata(name);
		*shape_rank = metadata.shape.size();
		*dtype = static_cast<int>(metadata.dtype);
		if (shape_capacity < metadata.shape.size() || (!shape && !metadata.shape.empty())) {
			setError(handle, "Shape buffer is too small");
			return 2;
		}
		for (size_t i = 0; i < metadata.shape.size(); ++i)
			shape[i] = metadata.shape[i];
		return 0;
	} catch (...) {
		setCurrentException(handle);
		return 3;
	}
}

int64_t php2xai_hdf5_count(PHP2XAIHDF5_Handle* handle)
{
	clearError(handle);
	if (!validHandle(handle))
		return -1;
	try {
		return handle->dataset->count();
	} catch (...) {
		setCurrentException(handle);
		return -1;
	}
}

int php2xai_hdf5_add(PHP2XAIHDF5_Handle* handle, const char* field,
	const void* data)
{
	clearError(handle);
	if (!validHandle(handle) || !field || !data) {
		setError(handle, "Invalid add arguments");
		return 1;
	}
	try {
		handle->dataset->add(field, data);
		return 0;
	} catch (...) {
		setCurrentException(handle);
		return 2;
	}
}

int php2xai_hdf5_read_indices(PHP2XAIHDF5_Handle* handle, const char* field,
	const int64_t* indices, size_t indices_count, void* output)
{
	clearError(handle);
	if (!validHandle(handle) || !field || (indices_count && !indices) || !output) {
		setError(handle, "Invalid read_indices arguments");
		return 1;
	}
	try {
		handle->dataset->readIndices(field,
			indices_count ? std::vector<int64_t>(indices, indices + indices_count)
						  : std::vector<int64_t>(), output);
		return 0;
	} catch (...) {
		setCurrentException(handle);
		return 2;
	}
}

const char* php2xai_hdf5_last_error(PHP2XAIHDF5_Handle* handle)
{
	return handle ? handle->error.c_str() : globalError.c_str();
}

}
