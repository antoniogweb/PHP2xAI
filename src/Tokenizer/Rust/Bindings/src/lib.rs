use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;
use std::slice;

use tokenizers::Tokenizer;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const {
        RefCell::new(None)
    };
}

/// Handle opaco esposto attraverso la C ABI.
///
/// Il codice C/PHP non accede direttamente ai campi interni.
#[repr(C)]
pub struct PHP2xAITokenizer {
    tokenizer: Tokenizer,
}

/// Risultato opaco di un'operazione di encoding.
#[repr(C)]
pub struct PHP2xAIEncoding {
    ids: *mut u32,
    length: usize,
}

fn set_last_error(message: impl ToString) {
    let message = message.to_string().replace('\0', " ");

    LAST_ERROR.with(|error| {
        *error.borrow_mut() = CString::new(message).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|error| {
        *error.borrow_mut() = None;
    });
}

/// Converte una stringa C obbligatoria in una stringa Rust UTF-8.
///
/// # Safety
///
/// `value` deve puntare a una stringa C valida e terminata da `\0`.
unsafe fn required_string<'a>(
    value: *const c_char,
    argument_name: &str,
) -> Result<&'a str, String> {
    if value.is_null() {
        return Err(format!("{argument_name} cannot be NULL"));
    }

    let c_string = unsafe { CStr::from_ptr(value) };

    c_string
        .to_str()
        .map_err(|error| format!("{argument_name} is not valid UTF-8: {error}"))
}

/// Restituisce l'ultimo errore prodotto nel thread corrente.
///
/// Il puntatore restituito appartiene alla libreria e non deve essere liberato.
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_last_error() -> *const c_char {
    LAST_ERROR.with(|error| {
        error
            .borrow()
            .as_ref()
            .map_or(ptr::null(), |message| message.as_ptr())
    })
}

/// Carica un tokenizer da un file tokenizer.json.
///
/// Restituisce NULL in caso di errore.
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_load(
    tokenizer_json_path: *const c_char,
) -> *mut PHP2xAITokenizer {
    clear_last_error();

    let result = (|| {
        let path = unsafe {
            required_string(
                tokenizer_json_path,
                "tokenizer_json_path",
            )?
        };

        let tokenizer = Tokenizer::from_file(path)
            .map_err(|error| {
                format!("Cannot load tokenizer from '{path}': {error}")
            })?;

        Ok::<PHP2xAITokenizer, String>(
            PHP2xAITokenizer { tokenizer },
        )
    })();

    match result {
        Ok(handle) => Box::into_raw(Box::new(handle)),

        Err(error) => {
            set_last_error(error);
            ptr::null_mut()
        }
    }
}

/// Libera un tokenizer creato con php2xai_tokenizer_load().
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_free(
    handle: *mut PHP2xAITokenizer,
) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

/// Codifica una stringa UTF-8 in una sequenza di token ID.
///
/// Restituisce NULL in caso di errore.
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_encode(
    handle: *const PHP2xAITokenizer,
    text: *const c_char,
    add_special_tokens: bool,
) -> *mut PHP2xAIEncoding {
    clear_last_error();

    let result = (|| {
        if handle.is_null() {
            return Err(
                "Tokenizer handle cannot be NULL".to_string(),
            );
        }

        let handle = unsafe { &*handle };

        let text = unsafe {
            required_string(text, "text")?
        };

        let encoding = handle
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|error| {
                format!("Cannot encode text: {error}")
            })?;

        /*
         * Utilizziamo un boxed slice anziché conservare direttamente un Vec.
         * In questo modo la capacità dell'allocazione coincide sempre con
         * la sua lunghezza e può essere ricostruita correttamente in fase
         * di rilascio.
         */
        let mut ids = encoding
            .get_ids()
            .to_vec()
            .into_boxed_slice();

        let length = ids.len();
        let ids_pointer = ids.as_mut_ptr();

        /*
         * La proprietà dell'allocazione passa a PHP2xAIEncoding.
         * Sarà recuperata da php2xai_tokenizer_encoding_free().
         */
        std::mem::forget(ids);

        Ok::<PHP2xAIEncoding, String>(
            PHP2xAIEncoding {
                ids: ids_pointer,
                length,
            },
        )
    })();

    match result {
        Ok(encoding) => Box::into_raw(Box::new(encoding)),

        Err(error) => {
            set_last_error(error);
            ptr::null_mut()
        }
    }
}

/// Restituisce il puntatore all'array interno degli ID.
///
/// Il puntatore resta valido finché non viene chiamata
/// php2xai_tokenizer_encoding_free().
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_encoding_ids(
    encoding: *const PHP2xAIEncoding,
) -> *const u32 {
    if encoding.is_null() {
        return ptr::null();
    }

    unsafe { (*encoding).ids }
}

/// Restituisce il numero di token ID presenti nell'encoding.
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_encoding_length(
    encoding: *const PHP2xAIEncoding,
) -> usize {
    if encoding.is_null() {
        return 0;
    }

    unsafe { (*encoding).length }
}

/// Libera un encoding creato con php2xai_tokenizer_encode().
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_encoding_free(
    encoding: *mut PHP2xAIEncoding,
) {
    if encoding.is_null() {
        return;
    }

    let encoding = unsafe {
        Box::from_raw(encoding)
    };

    if encoding.ids.is_null() {
        return;
    }

    /*
     * Ricostruiamo il boxed slice originariamente creato durante encode().
     */
    let slice_pointer = ptr::slice_from_raw_parts_mut(
        encoding.ids,
        encoding.length,
    );

    unsafe {
        drop(Box::from_raw(slice_pointer));
    }
}

/// Decodifica una sequenza di token ID in una stringa UTF-8.
///
/// La stringa restituita deve essere liberata chiamando
/// php2xai_tokenizer_string_free().
///
/// Restituisce NULL in caso di errore.
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_decode(
    handle: *const PHP2xAITokenizer,
    ids: *const u32,
    length: usize,
    skip_special_tokens: bool,
) -> *mut c_char {
    clear_last_error();

    let result = (|| {
        if handle.is_null() {
            return Err(
                "Tokenizer handle cannot be NULL".to_string(),
            );
        }

        if ids.is_null() && length > 0 {
            return Err(
                "ids cannot be NULL when length is greater than 0"
                    .to_string(),
            );
        }

        let handle = unsafe { &*handle };

        let ids: &[u32] = if length == 0 {
            &[]
        } else {
            unsafe {
                slice::from_raw_parts(ids, length)
            }
        };

        let decoded = handle
            .tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|error| {
                format!("Cannot decode token IDs: {error}")
            })?;

        CString::new(decoded).map_err(|_| {
            "Decoded text contains a null byte".to_string()
        })
    })();

    match result {
        Ok(decoded) => decoded.into_raw(),

        Err(error) => {
            set_last_error(error);
            ptr::null_mut()
        }
    }
}

/// Libera una stringa creata con php2xai_tokenizer_decode().
#[unsafe(no_mangle)]
pub extern "C" fn php2xai_tokenizer_string_free(
    value: *mut c_char,
) {
    if value.is_null() {
        return;
    }

    unsafe {
        drop(CString::from_raw(value));
    }
}