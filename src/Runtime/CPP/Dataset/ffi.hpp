#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PHP2XAIHDF5_Handle PHP2XAIHDF5_Handle;

enum PHP2XAIHDF5_DType {
	PHP2XAIHDF5_FLOAT32 = 1,
	PHP2XAIHDF5_FLOAT64 = 2,
	PHP2XAIHDF5_INT32 = 3,
	PHP2XAIHDF5_INT64 = 4
};

PHP2XAIHDF5_Handle* php2xai_hdf5_create(const char* filename);
PHP2XAIHDF5_Handle* php2xai_hdf5_open(const char* filename);
void php2xai_hdf5_destroy(PHP2XAIHDF5_Handle* handle);

int php2xai_hdf5_set_field(PHP2XAIHDF5_Handle* handle, const char* name,
	int dtype, const int64_t* shape, size_t rank);
int php2xai_hdf5_field_metadata(PHP2XAIHDF5_Handle* handle, const char* name,
	int* dtype, int64_t* shape, size_t shape_capacity, size_t* shape_rank);
int64_t php2xai_hdf5_count(PHP2XAIHDF5_Handle* handle);
int php2xai_hdf5_add(PHP2XAIHDF5_Handle* handle, const char* field,
	const void* data);
int php2xai_hdf5_read_indices(PHP2XAIHDF5_Handle* handle, const char* field,
	const int64_t* indices, size_t indices_count, void* output);

/* Returns the last error for this handle, or a thread-local error for failed
 * create/open calls. The returned string remains valid until the next FFI call
 * on the same thread. */
const char* php2xai_hdf5_last_error(PHP2XAIHDF5_Handle* handle);

#ifdef __cplusplus
}
#endif
