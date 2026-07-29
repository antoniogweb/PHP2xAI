#ifndef PHP2XAI_TOKENIZERS_H
#define PHP2XAI_TOKENIZERS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Tipi opachi.
 *
 * La struttura interna è gestita dalla libreria Rust e non è visibile
 * al codice C, C++ o PHP FFI.
 */
typedef struct PHP2xAITokenizer PHP2xAITokenizer;
typedef struct PHP2xAIEncoding PHP2xAIEncoding;

/*
 * Restituisce l'ultimo errore generato nel thread corrente.
 *
 * Il puntatore restituito appartiene alla libreria e non deve essere
 * liberato.
 *
 * Rimane valido fino alla successiva chiamata che modifica lo stato
 * dell'errore.
 *
 * Restituisce NULL quando non è disponibile alcun errore.
 */
const char *php2xai_tokenizer_last_error(void);

/*
 * Carica un tokenizer da un file tokenizer.json.
 *
 * Restituisce:
 * - un handle valido in caso di successo;
 * - NULL in caso di errore.
 *
 * In caso di errore, il relativo messaggio può essere ottenuto tramite
 * php2xai_tokenizer_last_error().
 */
PHP2xAITokenizer *php2xai_tokenizer_load(
    const char *tokenizer_json_path
);

/*
 * Libera un tokenizer creato da php2xai_tokenizer_load().
 *
 * Accetta anche NULL.
 */
void php2xai_tokenizer_free(
    PHP2xAITokenizer *handle
);

/*
 * Codifica una stringa UTF-8 in una sequenza di token ID.
 *
 * add_special_tokens:
 * - true: aggiunge gli eventuali token speciali configurati;
 * - false: non aggiunge token speciali.
 *
 * Restituisce:
 * - un encoding valido in caso di successo;
 * - NULL in caso di errore.
 *
 * L'encoding restituito deve essere liberato chiamando
 * php2xai_tokenizer_encoding_free().
 */
PHP2xAIEncoding *php2xai_tokenizer_encode(
    const PHP2xAITokenizer *handle,
    const char *text,
    bool add_special_tokens
);

/*
 * Restituisce un puntatore all'array interno degli ID.
 *
 * Il puntatore:
 * - appartiene all'encoding;
 * - non deve essere liberato direttamente;
 * - resta valido fino alla chiamata di
 *   php2xai_tokenizer_encoding_free().
 */
const uint32_t *php2xai_tokenizer_encoding_ids(
    const PHP2xAIEncoding *encoding
);

/*
 * Restituisce il numero di token ID presenti nell'encoding.
 *
 * Restituisce 0 se encoding è NULL.
 */
size_t php2xai_tokenizer_encoding_length(
    const PHP2xAIEncoding *encoding
);

/*
 * Libera un encoding creato da php2xai_tokenizer_encode().
 *
 * Libera anche l'array interno degli ID.
 * Accetta anche NULL.
 */
void php2xai_tokenizer_encoding_free(
    PHP2xAIEncoding *encoding
);

/*
 * Decodifica un array di token ID in una stringa UTF-8.
 *
 * ids:
 * - puntatore a un array di uint32_t;
 * - può essere NULL soltanto quando length è 0.
 *
 * skip_special_tokens:
 * - true: rimuove i token speciali dal risultato;
 * - false: mantiene i token speciali.
 *
 * Restituisce:
 * - una nuova stringa C in caso di successo;
 * - NULL in caso di errore.
 *
 * La stringa restituita deve essere liberata chiamando
 * php2xai_tokenizer_string_free().
 */
char *php2xai_tokenizer_decode(
    const PHP2xAITokenizer *handle,
    const uint32_t *ids,
    size_t length,
    bool skip_special_tokens
);

/*
 * Libera una stringa creata da php2xai_tokenizer_decode().
 *
 * Accetta anche NULL.
 */
void php2xai_tokenizer_string_free(
    char *value
);

#ifdef __cplusplus
}
#endif

#endif /* PHP2XAI_TOKENIZERS_H */