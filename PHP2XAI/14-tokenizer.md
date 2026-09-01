# Tokenizer BPE di PHP2xAI

## Credits e attribuzione

Il tokenizer di PHP2xAI è costruito sulla libreria open source **Hugging Face Tokenizers**, che fornisce le implementazioni efficienti del modello BPE, della tokenizzazione ByteLevel, della normalizzazione e della gestione del vocabolario. Il trainer Rust incluso in questo repository utilizza direttamente il crate `tokenizers` di Hugging Face; il file `tokenizer.json` generato segue il formato del tokenizer serializzato dalla libreria.

Il progetto riconosce e ringrazia Hugging Face e tutti i contributori del progetto Tokenizers:

- progetto: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers);
- documentazione: [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers);
- licenza e copyright: il codice upstream è distribuito con Apache License 2.0; consultare il file `ThirdParty/tokenizers/LICENSE` e gli avvisi presenti nel progetto upstream.

PHP2xAI non è affiliato, sponsorizzato o approvato da Hugging Face. Hugging Face e Tokenizers sono marchi e progetti dei rispettivi titolari. Il codice di Hugging Face Tokenizers resta soggetto alla propria licenza; questa attribuzione non modifica i termini della licenza upstream né quelli della licenza di PHP2xAI.

Per la configurazione concreta usata dal trainer, vedere il sorgente in `src/Tokenizer/Rust/Trainer/src/main.rs`: il tokenizer combina BPE con pre-tokenizzazione e decodifica ByteLevel, normalizzazione NFC e i token speciali `[PAD]`, `[UNK]`, `[BOS]` e `[EOS]`.


## Scopo

PHP2xAI usa un tokenizer ByteLevel BPE per convertire il testo in una sequenza di ID interi. Gli ID sono gli indici della tabella degli embedding: se la tabella ha forma `[V, D]`, la prima dimensione `V` deve essere la dimensione effettiva del vocabolario.

Questa guida spiega come generare il tokenizer usando il binario Linux incluso nel repository:

```text
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer
```

Il file generato si chiama `tokenizer.json`.

## Flusso completo

1. Preparare un corpus UTF-8, normalmente chiamato `corpus.txt`.
2. Eseguire il trainer Rust.
3. Conservare insieme `tokenizer.json` e `tokenizer.meta.json`.
4. Leggere `vocabulary_size` dai metadati.
5. Usare quel valore come `V` nella tabella `[V, D]`.
6. Caricare lo stesso `tokenizer.json` durante training e inferenza.

La mappatura token → ID è parte del modello. Sostituire il tokenizer dopo aver creato gli embedding può rendere incompatibili tutti i pesi della tabella.

## Corpus

Il corpus è un file di testo UTF-8. Ogni riga è un campione:

```text
Questo è il primo documento.
Il tokenizer impara dai testi presenti nel corpus.
Una riga può contenere una frase o un documento lungo.
```

Le righe vuote e quelle composte solo da spazi vengono ignorate nel calcolo delle statistiche. Il corpus dovrebbe essere rappresentativo dei dati usati dal modello: includere maiuscole, punteggiatura, numeri, accenti ed emoji quando fanno parte dei dati reali.

Non occorre inserire nel corpus gli ID dei token o i token speciali. È il trainer a registrarli.

Controllare la codifica prima dell’esecuzione. Per esempio:

```bash
file -bi /percorso/dati/corpus.txt
```

## Prerequisiti Linux

Il binario incluso è compilato per Linux `x86_64`:

```bash
uname -m
file src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer
```

Il primo comando dovrebbe restituire `x86_64`. Se necessario, rendere eseguibile il file:

```bash
chmod +x src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer
```

## Generazione minima

Dalla root del repository:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input /percorso/dati/corpus.txt
```

Senza `--output`, il trainer salva nella directory corrente (`pwd`):

```text
tokenizer.json
tokenizer.meta.json
```

La directory corrente non è la directory del binario. Per evitare ambiguità si può lavorare in una directory dedicata:

```bash
mkdir -p artifacts/tokenizer
cd artifacts/tokenizer

/percorso/PHP2xAI/src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input /percorso/dati/corpus.txt
```

## Percorso esplicito

Con `--output` (o `-o`) si sceglie il nome del tokenizer:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input /percorso/dati/corpus.txt \
    --output /percorso/modello/tokenizer.json
```

Le directory genitore mancanti vengono create automaticamente. Il metadato viene scritto accanto al tokenizer con nome fisso:

```text
/percorso/modello/tokenizer.json
/percorso/modello/tokenizer.meta.json
```

Usare l’estensione `.json`, non `.js`.

## Opzioni

```text
-i, --input <FILE>
    Corpus UTF-8. Obbligatorio.

-o, --output <FILE>
    Destinazione del tokenizer JSON.
    Default: tokenizer.json nella directory corrente.

-v, --vocab-size <NUMBER>
    Dimensione massima del vocabolario.
    Default: 30000.

-f, --min-frequency <NUMBER>
    Frequenza minima delle coppie BPE.
    Default: 2.

--pretty
    Scrive JSON formattato invece del formato compatto.

-h, --help
    Mostra l’aiuto.
```

Per vedere la versione dell’aiuto effettivamente incorporata nel binario:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer --help
```

### Limite massimo del vocabolario

`--vocab-size 30000` è un limite massimo, non una promessa di 30.000 token. Un corpus piccolo o una frequenza minima alta può produrre un vocabolario più piccolo. Per la tabella degli embedding usare sempre `vocabulary_size` del file meta, mai copiare alla cieca il valore passato con `--vocab-size`.

Esempio completo:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input corpus.txt \
    --output model/tokenizer.json \
    --vocab-size 30000 \
    --min-frequency 2 \
    --pretty
```

## File prodotti

### `tokenizer.json`

Contiene il tokenizer completo: modello BPE, vocabolario token → ID, merge BPE, normalizzatore Unicode NFC, pre-tokenizer ByteLevel, decoder ByteLevel e token speciali. È il file passato al binding Rust/C e alla classe tokenizer PHP.

### `tokenizer.meta.json`

Contiene statistiche calcolate dopo l’addestramento. Un esempio è:

```json
{
  "vocabulary_size": 18472,
  "samples": 50000,
  "max_length": 3105,
  "average_length": 291.13634,
  "p50": 216,
  "p95": 750,
  "p99": 1139
}
```

Il campo usato dal modello è:

```json
"vocabulary_size": 18472
```

Nel trainer attuale gli ID sono contigui a partire da zero, quindi:

```text
ID validi: 0 ... vocabulary_size - 1
embedding table: [vocabulary_size, D]
```

Le altre proprietà significano:

- `samples`: numero di righe non vuote;
- `max_length`: lunghezza massima in token;
- `average_length`: lunghezza media;
- `p50`, `p95`, `p99`: percentili delle lunghezze;
- `vocabulary_size`: numero di righe necessarie nella tabella degli embedding.

## Token speciali

Il trainer assegna gli ID in questo ordine:

```text
0 -> [PAD]
1 -> [UNK]
2 -> [BOS]
3 -> [EOS]
```

`[PAD]` riempie le sequenze, `[UNK]` rappresenta contenuti sconosciuti, `[BOS]` indica l’inizio e `[EOS]` la fine. Sono già inclusi in `vocabulary_size`: non aggiungere `+4`.

Per esempio, se il meta riporta `30000`, la tabella corretta è `[30000, D]`, con righe da `0` a `29999`; le prime quattro righe sono quelle speciali.

## Usare i metadati in PHP

Il modello deve leggere `vocabulary_size` dal file generato:

```php
<?php

$metadataPath = __DIR__ . '/tokenizer.meta.json';
$json = file_get_contents($metadataPath);
if ($json === false) {
    throw new RuntimeException("Impossibile leggere {$metadataPath}");
}

$metadata = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
$V = (int) ($metadata['vocabulary_size'] ?? 0);
if ($V < 4) {
    throw new RuntimeException('vocabulary_size non valido');
}

$D = 512;
$embTable = Tensor::init([$V, $D], 0.05);
```
