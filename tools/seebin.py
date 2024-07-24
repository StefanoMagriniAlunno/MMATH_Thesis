# mostro il file passato come primo argomento
# il file passato è binario, l'obiettivo è mostrare il file sorgente come una tabella
# se passo "b" come secondo argomento, mostro una tabella dove ogni argomento è un byte
# se passo "h" come secondo argomento, mostro una tabella dove ogni argomento è un byte in esadecimale (hex = 2 byte)
# se passo "w" come secondo argomento, mostro una tabella dove ogni argomento è un byte in esadecimale (word = 4 byte)
# se passo "d" come secondo argomento, mostro una tabella dove ogni argomento è un byte in esadecimale (dword = 8 byte)
# se passo "q" come secondo argomento, mostro una tabella dove ogni argomento è un byte in esadecimale (qword = 16 byte)
# inoltre come terzo argomento passo il numero di colonne da mostrare

import sys

# leggo gli argomenti (file, tipo, colonne, righe, output)
if len(sys.argv) != 6:
    print("Usage: python3 seebin.py <file> <type> <columns> <rows> <output>")
    sys.exit(1)

file = sys.argv[1]
type = sys.argv[2]
columns = int(sys.argv[3])
rows = int(sys.argv[4])
output = sys.argv[5]

# leggo il file binario
with open(file, "rb") as f:
    data = f.read()

print(f"letti {len(data)} byte")

# mostro il file come tabella
if type == "b":
    string = "\n".join(
        [
            " ".join(f"{x:02x}" for x in data[i : i + columns * 1])
            for i in range(0, min(len(data), columns * rows), columns)
        ]
    )
elif type == "h":
    string = "\n".join(
        [
            " ".join(f"{x:02x}" for x in data[i : i + columns * 2])
            for i in range(0, min(len(data), columns * rows), columns)
        ]
    )
elif type == "w":
    string = "\n".join(
        [
            " ".join(f"{x:02x}" for x in data[i : i + columns * 4])
            for i in range(0, min(len(data), columns * rows), columns)
        ]
    )
elif type == "d":
    string = "\n".join(
        [
            " ".join(f"{x:02x}" for x in data[i : i + columns * 8])
            for i in range(0, min(len(data), columns * rows), columns)
        ]
    )
elif type == "q":
    string = "\n".join(
        [
            " ".join(f"{x:02x}" for x in data[i : i + columns * 16])
            for i in range(0, min(len(data), columns * rows), columns)
        ]
    )
else:
    print("Invalid type")
    sys.exit(1)

print(f"formattati {len(string)} byte")

with open(output, "w") as f:
    f.write(string)
