# Gleiche Frage, andere Antwort: Kann man ein LLM deterministisch betreiben?

Temperatur null, fester Seed, identischer Prompt. Und trotzdem eine andere Antwort. Ich wollte wissen, woran das liegt, und habe es zusammen mit Claude nachgemessen.

## Die Frage

Auf einem Meetup zu angewandter KI kam beim Networking eine Diskussion auf, die mich nicht losgelassen hat: Kann man ein großes Sprachmodell dazu bringen, jedes Mal exakt dieselbe Antwort zu geben?

Die Frage war nicht akademisch. Einer aus der Runde arbeitete in der Finanzbranche. Automatisch erzeugte Ergebnisse müssen dort für Prüfungen nachvollziehbar sein. „Haben wir probiert“, sagte er. „Temperatur null, fester Seed, identischer Prompt. Trotzdem verschiedene Antworten.“

Dagegen stand ein einfaches Argument: Ein LLM ist im Kern eine Kette von Matrixmultiplikationen. Gleiche Eingabe, gleiche Gewichte, gleiche Parameter. Mathematik ist Mathematik, da muss jedes Mal dasselbe herauskommen.

Ich war auf der Seite der Mathematik, konnte mir seine Beobachtung aber nicht erklären. Also haben wir es ausprobiert, Claude und ich: auf eigener Hardware, mit offenen Modellen, in über 3.000 Messläufen.

Den Plan habe ich vorgegeben: erst der Normalbetrieb, dann Temperatur null, dann den möglichen Ursachen einzeln nachgehen. Unterwegs kamen die Ideen mal von mir, mal von Claude. Claude hat die Messskripte geschrieben, die Läufe ausgewertet und die Quellen mitrecherchiert. Die Gegenproben mit einem zweiten Modell und auf einer NVIDIA-Karte habe ich dazugenommen. Was in diesen Text kommt, habe ich entschieden. Wenn im Folgenden „wir“ steht, sind Claude und ich gemeint.

Vorweg: Beide Seiten hatten recht. Warum das so ist und was man tun kann, zeigen wir Schritt für Schritt, jede mögliche Ursache mit einer eigenen Messung. Eine davon, das „Gedächtnis“ des Servers, konnten wir eingrenzen, aber nicht bis ins Letzte aufklären.

## Was die Forschung sagt

Neu ist die Frage nicht. **Thinking Machines** schickte 2025 tausendmal dieselbe Anfrage mit Temperatur null an ein großes Qwen3-Modell und bekam 80 verschiedene Antworten. **Yuan et al.** (2025) zeigen, dass schon die Zahl der gemeinsam verarbeiteten Anfragen, die Zahl der GPUs oder der GPU-Typ die Antworten verändern. **Ingonyama** (2024) hat llama.cpp so umgebaut, dass es auf verschiedenen NVIDIA-GPUs identische Ausgaben liefert. Dafür fiel die Verarbeitung des Prompts (gemessen auf einer RTX 4080) von 302 auf 43 Tokens pro Sekunde.

Uns hat interessiert, ob sich das auf eigener Hardware nachvollziehen lässt und welche Ursache wie viel ausmacht.

## Der Versuchsaufbau

- **Hardware:** ein Mini-PC mit AMD Strix Halo (Grafikeinheit Radeon 8060S) und 128 GB Arbeitsspeicher
- **Software:** llama.cpp, eine verbreitete Open-Source-Software, um Sprachmodelle lokal zu betreiben. Sie lief als Server, so wie ihn auch eine Anwendung ansprechen würde, und rechnete über die Grafikschnittstelle Vulkan, mit eingeschaltetem Flash Attention.
- **Modell:** Qwen3-8B, ein offenes Modell mit 8 Milliarden Parametern, auf 4 Bit komprimiert (Q4_K_M)
- **Aufgabe:** „Write a short story about a lighthouse keeper who finds a strange message in a bottle. Be creative.“, jeweils 200 Tokens Antwort

Warum eine Geschichte? Bei kreativen Aufgaben gibt es oft mehrere fast gleich gute Fortsetzungen. Schon beim ersten Wort schätzt das Modell „On“ auf 55 % und „\*\*“ (den Beginn einer fett gesetzten Überschrift) auf 40 %. Solche knappen Entscheidungen sind genau die Stellen, an denen Abweichungen sichtbar werden.

„Gleiche Antwort“ heißt in allen Tests: derselbe Text, Zeichen für Zeichen. Pro Test haben wir die Anfrage 30-mal gestellt, wenn nicht anders angegeben. Der Server lief eigens für diese Tests, ohne andere Nutzer. Gleichzeitige Anfragen gab es nur dort, wo wir sie absichtlich dazugeschaltet haben. Den Zwischenspeicher für bereits verarbeitete Prompts, der bei llama.cpp normalerweise eingeschaltet ist, haben wir nur in Schritt 6 und für eine Vergleichsfolge in Schritt 7 eingeschaltet.

Die genauen Zahlen gelten nur für dieses Setup. Zwei Gegenproben nach Schritt 7 prüfen, ob die Befunde auch mit einem anderen Modell und einer anderen Software gelten.

## Schritt 1: Temperatur 0,7 – der Normalfall

Ein Sprachmodell schreibt Token für Token. Für jeden Schritt berechnet es eine Wahrscheinlichkeit für jedes mögliche nächste Token, dann wird eines davon ausgewählt. Die Temperatur steuert, wie diese Auswahl ausfällt: Bei niedriger Temperatur gewinnt fast immer der Favorit, bei höherer kommen auch weniger wahrscheinliche Kandidaten zum Zug. Bei Temperatur null wird immer das wahrscheinlichste Token genommen.

0,7 ist der Wert, den die Entwickler von Qwen3 für dieses Modell empfehlen. Ohne festen Seed holt sich die Software für jede Anfrage einen neuen Zufallswert.

**Ergebnis: 30 Anfragen, 30 verschiedene Antworten.** Meist trennen sich die Wege schon in den ersten zwei, drei Tokens; das längste gemeinsame Anfangsstück zweier Antworten war 26 Tokens lang. Mal beginnt die Geschichte mit einer Überschrift wie „The Keeper of the Hollow Light“, mal direkt mit „On the stormy coast of Northumberland …“ oder „On the windswept cliffs of Seagull Point …“.

Das ist kein Fehler, sondern genau so gewollt. Für einen Chatbot oder eine kreative Aufgabe ist Abwechslung erwünscht. Für ein Audit ist das natürlich nichts.

## Schritt 2: Fester Seed – fast, aber nicht ganz

Der „Zufall“ beim Auswählen der Tokens kommt aus einem Zufallsgenerator, genauer: einem Pseudo-Zufallsgenerator. Das ist eine Formel, die aus einem Startwert, dem Seed, eine feste Zahlenfolge berechnet. Gleicher Seed, gleiche Folge, wie eine vorgedruckte Liste von Würfelergebnissen. Also: Seed auf 42 festlegen, Temperatur bleibt bei 0,7.

**Ergebnis: 24 von 30 Antworten identisch.** Die übrigen sechs weichen erst spät ab, zwischen dem 78. und dem 173. Token, also mitten in der Geschichte.

Zwei Details fallen auf. Erstens kamen alle sechs Abweichungen in den ersten sieben Läufen, danach 23-mal hintereinander dieselbe Antwort. Zweitens liefert ein frisch gestarteter Server mit demselben Seed eine weitere Variante, die in der Messreihe gar nicht vorkam. Die ist aber bei fünf Neustarts fünfmal exakt gleich.

Wenn die Würfelergebnisse jedes Mal gleich sind, die Antworten aber trotzdem nicht, dann muss sich etwas anderes geändert haben: die Wahrscheinlichkeiten selbst, aus denen gewürfelt wird. Liegt eine Zufallszahl knapp an der Grenze zwischen zwei Tokens, reicht eine winzige Verschiebung, und ein anderes Token gewinnt. Woher solche Verschiebungen kommen, zeigt der nächste Schritt.

## Schritt 3: Temperatur null – kein Zufall mehr, trotzdem verschieden

Bei Temperatur null wird nicht mehr gewürfelt. Das Modell nimmt in jedem Schritt das wahrscheinlichste Token, der Seed spielt keine Rolle mehr. Das ist die Einstellung, die der Gesprächspartner vom Meetup verwendet hatte.

**Ergebnis: 28 von 30 Antworten identisch, insgesamt drei Varianten.** Seltener als bei Schritt 2, aber es passiert. Die erste Weiche liegt beim 103. Token:

> „…its light had dimmed with time. Elias **had** long since stopped using the original lamp …“

> „…its light had dimmed with time. Elias **kept** it running, not for the tourists or the ships …“

An dieser Stelle hat das Modell zwei fast gleich starke Kandidaten: „had“ mit 42,7 % und „kept“ mit 41,9 %. Die beiden trennen 0,8 Prozentpunkte. Wenn sich die Rechnung minimal verschiebt, wechselt der Favorit, und ab da schreibt das Modell eine andere Geschichte. Die dritte Variante zweigt zehn Tokens später an einer weiteren knappen Stelle ab: „coast“ mit 18,2 % gegen „ships“ mit 17,7 %.

**Warum sollte sich eine deterministische Rechnung überhaupt verschieben?** Wegen der Art, wie Computer mit Kommazahlen rechnen. Sie speichern Zahlen mit begrenzter Genauigkeit und runden nach jeder Operation. Dadurch hängt das Ergebnis von der Reihenfolge ab. Ein Beispiel im verbreiteten 32-Bit-Gleitkommaformat, in dem sich Zahlen um 100 Millionen nur in Achterschritten darstellen lassen:

- (100.000.000 + 3) + 3 = **100.000.000**, weil jede 3 für sich weggerundet wird
- 100.000.000 + (3 + 3) = **100.000.008**, weil die 6 auf die nächste darstellbare Zahl aufgerundet wird

Gleiche Zahlen, andere Klammern, anderes Ergebnis. Wie stark das Deep-Learning-Programme betrifft, haben Shanmugavelu et al. (2024) untersucht. Ein Sprachmodell mit 8 Milliarden Parametern rechnet für jedes Token grob 8 Milliarden Multiplikationen und Additionen. Die Grafikkarte verteilt sie auf viele Rechenkerne und fasst die Teilsummen danach zusammen. Wird anders aufgeteilt, wird anders gerundet, und am Ende steht eine minimal andere Wahrscheinlichkeit. Meistens ist das egal. An Stellen wie Token 103 nicht.

Am Rande: Temperatur null ist bei Qwen3 nicht für jeden Betrieb vorgesehen. Für den Thinking-Modus steht in der Modellbeschreibung: „DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.“ Wir haben ohne Thinking getestet; für diesen Modus nennt die Modellbeschreibung nur 0,7 als empfohlene Temperatur.

Bleibt die eigentliche Frage: Warum rechnet derselbe Rechner mit demselben Modell überhaupt einmal so und einmal anders? Dem sind wir Ursache für Ursache nachgegangen.

## Schritt 4: Würfelt die Grafikkarte?

Die naheliegende Erklärung lautet: Auf der Grafikkarte rechnen tausende Kerne parallel. Wer zuerst fertig ist, wird zuerst addiert, bei jedem Lauf in anderer Reihenfolge. Das klingt plausibel. Thinking Machines kommt allerdings zu dem Schluss, dass es für die Berechnung eines Sprachmodells typischerweise nicht zutrifft: Der übliche Rechenweg enthält keine Schritte, bei denen die Reihenfolge der Additionen davon abhängt, welcher Kern zuerst fertig ist.

Das lässt sich prüfen. Wir haben 30-mal nur den Prompt verarbeiten lassen und uns jeweils die Wahrscheinlichkeiten der drei besten Kandidaten für das erste Token ausgeben lassen, mit allen Nachkommastellen.

**Ergebnis: 30-mal exakt dieselben Werte, bis in die letzte Stelle.**

Auch die vollständige Generierung ist reproduzierbar, wenn die Ausgangslage gleich ist: Ein frisch gestarteter Server lieferte bei zehn Neustarts zehnmal dieselbe Geschichte, alle 200 Tokens.

Andere Zahlen gibt es nur, wenn sich der Rechenweg ändert. Ein Beispiel ist Flash Attention, ein alternativer Algorithmus für einen Teil der Rechnung. Schaltet man ihn ab, verschiebt sich in einem Test mit einem kürzeren Prompt die Wahrscheinlichkeit des Favoriten von 55,8 % auf 55,1 %. Aber auch dann kommen bei jedem Lauf dieselben Werte heraus.

Die Grafikkarte würfelt also nicht. Wenn die Antworten trotzdem variieren, muss sich der Rechenweg zwischen den Anfragen ändern. Was kann ihn ändern? Drei Kandidaten: andere Anfragen, die gleichzeitig laufen, der Zwischenspeicher für bereits verarbeitete Prompts und die Vorgeschichte des Servers.

## Schritt 5: Die anderen Nutzer

Ein Server bedient normalerweise viele Nutzer gleichzeitig. Statt die Anfragen nacheinander abzuarbeiten, fasst er sie zu einem Stapel (Batch) zusammen und rechnet sie gemeinsam. Das ist deutlich effizienter. Thinking Machines hat gezeigt, wo das Problem liegt: Je nach Größe des Stapels teilen die Rechenbausteine die Arbeit unterschiedlich auf. Gleiche Rechnung, andere Aufteilung, andere Rundung. Auch in llama.cpp hängt die Aufteilung laut Quellcode (Vulkan-Teil) von der Stapelgröße ab.

Der Test: ein Server mit vier Plätzen für gleichzeitige Anfragen, für jede Messreihe frisch gestartet. Einmal 30 Anfragen nacheinander. Einmal 30 Anfragen, bei denen jedes Mal drei weitere Anfragen gleichzeitig starten, so wie auf einem Server, den sich mehrere Nutzer teilen.

- **Allein:** Die häufigste Antwort kam 25 von 30 Mal, insgesamt gab es 6 Varianten. Ab Lauf 11 waren aber alle 20 Antworten identisch.
- **Mit drei parallelen Anfragen:** Die häufigste Antwort kam nur 15 von 30 Mal, ebenfalls 6 Varianten. Und auch ab Lauf 11 gab es noch 3 Varianten, die häufigste nur 11 von 20 Mal.

Ohne parallele Anfragen wackeln nur die ersten Läufe, dann ist die Antwort stabil. Warum das so ist, dazu gleich mehr. Mit parallelen Anfragen stabilisiert sie sich in 30 Läufen nicht.

Ganz sauber getrennt ist das hier allerdings nicht. Die fremden Anfragen gehören auch zur Vorgeschichte des Servers, und die spielt eine eigene Rolle (Schritt 7). Den Effekt des Stapels allein zeigt die Gegenprobe mit vLLM weiter unten.

Genau das ist die Lage bei einer Cloud-API. Dort teilt man sich die Server mit unzähligen anderen Nutzern, und wie die Anfragen gestapelt werden, sieht man von außen nicht.

Nebenbei: Schon die Zahl der Plätze verändert die Antwort. Ein frisch gestarteter Server mit einem Platz lieferte bei der allerersten Anfrage jedes Mal dieselbe Geschichte (15 von 15 Neustarts). Mit zwei oder vier Plätzen kam eine andere Geschichte, aber auch sie bei jedem Neustart dieselbe (je 5 Neustarts). Zwei und vier Plätze lieferten sogar dieselbe Geschichte. Die Konfiguration des Servers bestimmt also mit, welche Antwort man bekommt.

## Schritt 6: Der Zwischenspeicher – eine falsche Fährte

Server merken sich bereits verarbeitete Prompts in einem Zwischenspeicher, dem Cache. Kommt dieselbe Anfrage noch einmal, muss nur der Rest neu berechnet werden. Unsere Vermutung: Mal wird der Prompt komplett neu gerechnet, mal zum Teil aus dem Cache genommen, und diese unterschiedliche Aufteilung erzeugt unterschiedliche Rundungen.

**Ergebnis mit eingeschaltetem Cache: Die häufigste Antwort kam nur noch 15 von 30 Mal.** Und das Muster ist auffällig: Ab dem achten Lauf wechseln sich zwei Antworten streng ab, Lauf für Lauf.

Das sieht nach einer klaren Erklärung aus. Also haben wir nachgesehen, wie viel vom Prompt der Server bei jeder Anfrage tatsächlich neu berechnet: ab der zweiten Anfrage jedes Mal genau ein Token, die übrigen 32 kommen aus dem Cache. Immer gleich. Und trotzdem wechseln die Antworten.

Unsere Vermutung war also falsch. Der Cache verändert zwar das Muster, aber nicht dadurch, dass der Prompt mal so, mal so aufgeteilt wird. Etwas anderes ändert sich von Anfrage zu Anfrage. Die Frage ist: was?

## Schritt 7: Der Server hat ein Gedächtnis

Wenn es kein Zufall ist, müsste sich das Ganze wiederholen lassen. Also haben wir den Server frisch gestartet, ihm 16-mal dieselbe Anfrage geschickt und das Ganze dreimal wiederholt, jedes Mal nach einem Neustart. Jeder Buchstabe steht für eine der unterschiedlichen Geschichten:

```
Neustart 1:  A A B C D C E E E D D D D D D D
Neustart 2:  A A B C D C E E E D D D D D D D
Neustart 3:  A A B C D C E E E D D D D D D D
```

**Dreimal exakt dieselbe Folge.** Mit eingeschaltetem Cache ergibt sich ein anderes Muster, aber auch das ist bei jedem Neustart wieder identisch:

```
A A F A A B E E E F E F E F E F
```

E und F sind übrigens genau die beiden Antworten, die sich in Schritt 6 abgewechselt haben. Dort hatte der Server vorher schon andere Messreihen abgearbeitet. Deshalb der andere Anfang, aber dasselbe Ende.

Das ist der Kern der Sache. Bei Anfragen nacheinander ist der Server als Ganzes vollkommen deterministisch. Aber er hat ein Gedächtnis: Welche Antwort er gibt, hängt davon ab, was er vorher gerechnet hat. Die dritte Anfrage bekommt eine andere Geschichte als die erste, obwohl sie Zeichen für Zeichen gleich ist. Und zwar bei jedem Durchlauf wieder dieselbe andere.

Wo steckt dieses Gedächtnis? Zwei Gegenproben:

- **Den Platz der Anfrage auf dem Server vor jeder Anfrage leeren** (dafür bietet llama.cpp eine eigene Funktion): hilft nicht. 15 Anfragen, fünf verschiedene Antworten.
- **Für jede Anfrage ein neues Programm starten**, das das Modell lädt, die Antwort erzeugt und sich wieder beendet: 15 Anfragen, 15-mal dieselbe Antwort.

Das Gedächtnis steckt also nicht in den gespeicherten Inhalten, sondern irgendwo im Zustand des laufenden Programms. Unsere Vermutung, ausdrücklich nicht gemessen: Je nach Vorgeschichte landet eine Anfrage an einer anderen Stelle im internen Speicher des Servers. Dadurch ändert sich die Länge eines Zwischenergebnisses, und im Quellcode der Vulkan-Version hängt die Aufteilung der Rechnung genau von dieser Länge ab. Andere Aufteilung, andere Rundung, andere Geschichte.

Ganz verstanden haben wir es nicht. Zur Gegenprobe lief der Server nur auf dem Prozessor, ganz ohne Grafikkarte: 24 Anfragen nach zwei Neustarts, 24-mal dieselbe Antwort. In diesem Test zeigte sich das Gedächtnis also nur, wenn die Grafikkarte mitrechnete. Das passt zu unserer Vermutung, bewiesen ist sie damit nicht.

Auf dem Weg dorthin sind wir selbst in eine Falle getappt. Die übliche Einstellung, alle Schichten des Modells auf dem Prozessor zu rechnen, hält die Grafikkarte nicht vollständig heraus: Rechenschritte mit 32 oder mehr Tokens auf einmal lagert llama.cpp trotzdem dorthin aus, und unser Prompt hat 33. Im ersten Kontrolllauf zeigte sich das Gedächtnis deshalb ebenfalls. Erst mit ausdrücklich abgeschalteter Grafikkarte war es verschwunden.

Praktisch heißt das: Bei vielen Nutzern gehören deren Anfragen zur Vorgeschichte, und die kann niemand vorhersagen. Für den Einzelnen sieht das aus wie Zufall.

Damit lassen sich beide Seiten vom Meetup versöhnen: Die Mathematik ist deterministisch. Nur hängt das Ergebnis von mehr ab als vom eigenen Prompt.

## Gegenprobe: ein ganz anderes Modell

Ein Modell ist kein Beweis. Deshalb haben wir die Kernaussagen an einem zweiten Modell geprüft, das sich in fast allem unterscheidet: gpt-oss-120b von OpenAI. Mit 117 Milliarden Parametern ist es gut vierzehnmal so groß wie Qwen3-8B, anders komprimiert und anders gebaut. Es ist ein sogenanntes Mixture-of-Experts-Modell, bei dem für jedes Token nur ein Teil des Netzes rechnet. Gleicher Rechner, gleiche Software, gleiche Aufgabe, Temperatur null. Das Modell denkt vor der Antwort kurz nach. Das haben wir auf die niedrigste Stufe gestellt, dann ist es ein einziger Satz.

- **Laufender Server, 30 Anfragen nacheinander:** 26 verschiedene Antworten, die häufigste kam dreimal. Anders als bei Qwen pendelt sich nichts ein.
- **Frischer Server für jede Anfrage, 10 Anfragen:** 10-mal dieselbe Antwort.
- **16 Anfragen nach einem Neustart, dreimal wiederholt:** 16 verschiedene Antworten, aber bei allen drei Neustarts exakt dieselbe Folge. Es ist sogar dieselbe Folge wie bei den ersten 16 Anfragen des ersten Tests.

Die meisten Abweichungen beginnen schon im ersten Satz der Geschichte: „The lighthouse **at Grayhaven** had stood for a century …“ oder „The lighthouse **on Gull’s Point** had stood for a century …“.

Das Muster ist dasselbe wie bei Qwen, nur deutlich stärker: Der Server als Ganzes ist deterministisch, die einzelne Antwort hängt von seiner Vorgeschichte ab. Warum gpt-oss so viel empfindlicher reagiert, haben wir nicht untersucht. Größe, Kompression und Bauart unterscheiden sich gleichzeitig, das lässt sich mit diesem Test nicht auseinanderhalten.

## Gegenprobe: eine andere Software

Hat jeder Server so ein Gedächtnis? Und gibt es Software, die das Problem löst? vLLM, eine verbreitete Server-Software, hat dafür einen eigenen Modus: Er soll die Rechnung immer gleich aufteilen, egal wie viele andere Anfragen im selben Stapel stecken. Für meinen Mini-PC ist dieser Modus laut vLLM nicht vorgesehen, also kam ein zweiter Rechner dazu: ein PC mit einer NVIDIA RTX 4070. Darauf lief vLLM mit demselben Qwen3-8B, allerdings in einer anderen komprimierten Fassung (AWQ), die vLLM direkt unterstützt. Gleiche Aufgabe, Temperatur null.

Zuerst eine Überraschung: 216 Anfragen und immer dieselbe Geschichte. Nacheinander, mit drei parallelen Anfragen, nach Neustarts, mit und ohne den deterministischen Modus. Sogar beide Modi lieferten Zeichen für Zeichen denselben Text.

Das hat uns misstrauisch gemacht, denn die beiden Modi rechnen nachweislich mit unterschiedlichen Rechenbausteinen. Also haben wir nicht nur den Text verglichen, sondern die Wahrscheinlichkeit jedes einzelnen Tokens, bis in die letzte Stelle:

- **Nacheinander:** in beiden Modi 10 von 10 identisch.
- **Mit drei parallelen Anfragen:** Im normalen Modus verschieben sich die Zahlen in 9 von 10 Läufen bei rund 180 von 200 Tokens, im zehnten bei 81. Im deterministischen Modus 10 von 10 identisch.
- **Nach einem Neustart:** in beiden Modi identisch mit dem ersten Start.

Drei Erkenntnisse:

- **Gleicher Text heißt nicht gleiche Rechnung.** Im normalen Modus verschieben die parallelen Anfragen die Zahlen meist fast überall. Der Favorit hat dabei nie gewechselt. Ohne parallele Anfragen lag die knappste Stelle bei 34,9 % gegen 34,4 %. Mit parallelen Anfragen lagen die beiden Favoriten an zwei Stellen in einzelnen Läufen sogar exakt gleichauf (je 34,6 % und je 50,0 %). Bei einem Gleichstand muss sich das System für eines entscheiden, und das war in allen Läufen dasselbe Token. Das war also knapp. Mit einem anderen Prompt oder einer längeren Antwort kann das anders ausgehen.
- **Der deterministische Modus wirkt.** Mit ihm ändern parallele Anfragen nichts mehr, nicht einmal in der letzten Stelle (10 von 10 Läufen). Einer der beiden Gleichstände (50 % zu 50 %) besteht dort in jedem Lauf, entschieden wird immer gleich.
- **Das Gedächtnis aus Schritt 7 trat bei vLLM nicht auf.** Nacheinander gestellte Anfragen waren bit-identisch, auch ohne deterministischen Modus und obwohl der Zwischenspeicher für Prompts eingeschaltet war (bei vLLM der Standard). Es ist also keine Eigenschaft jedes LLM-Servers. Woran es liegt, lässt dieser Vergleich allein offen: Software, Grafikkarte (NVIDIA statt AMD) und Komprimierung (AWQ statt Q4_K_M) unterscheiden sich gleichzeitig. Der Nachtest unten grenzt es weiter ein.

Bezahlt wird das allerdings mit Zeit: Statt rund 2,5 Sekunden brauchte eine einzelne Anfrage im deterministischen Modus rund 30, etwa zwölfmal so lange. Thinking Machines nennt einen kleineren Aufschlag (55 statt 26 Sekunden), gemessen allerdings für 1.000 Anfragen zusammen und in einem anderen Aufbau. Die Zahlen lassen sich nicht direkt vergleichen.

**Nachtest mit einem kleinen Modell.** Zwei Fragen blieben offen: Woher kommt der große Aufschlag? Und liegt das fehlende Gedächtnis an der Software? Also haben wir mit einem dritten Modell nachgemessen, Qwen3-1.7B. Es ist klein genug, um es ohne Komprimierung zu rechnen. Damit rechneten alle mit denselben Gewichten: llama.cpp auf dem Mini-PC (über Vulkan), llama.cpp auf der RTX 4070 (über NVIDIAs Schnittstelle CUDA) und vLLM auf der RTX 4070 in beiden Modi. Insgesamt 2.264 Anfragen, diesmal fast alle mit der Wahrscheinlichkeit jedes Tokens.

- **Der Aufschlag:** Ohne Komprimierung brauchte der deterministische Modus 3,25 statt 1,62 Sekunden pro Anfrage, also das Doppelte. Das spricht dafür, dass die zwölffache Zeit vor allem an der Komprimierung liegt: Für AWQ-Modelle schaltet vLLM in diesem Modus seine schnellen Rechenbausteine ab, weil sie die Rechnung nicht immer gleich aufteilen. Bewiesen ist das nicht, das neue Modell ist ja auch kleiner.
- **Das Gedächtnis:** llama.cpp auf der RTX 4070 hatte keins. 416 Anfragen nacheinander und nach Neustarts, alle bis in die letzte Stelle gleich. Auf dem Mini-PC schon, auch ohne Komprimierung: Nach jedem Neustart unterscheiden sich die Wahrscheinlichkeiten der ersten vier Anfragen, ab der fünften bleiben sie gleich. Den Text hat das bei diesem Modell nicht verändert, keine Stelle war knapp genug. Am Programm als Ganzem und an der Komprimierung liegt das Gedächtnis also nicht. In unseren Tests trat es nur auf, wenn über Vulkan auf der AMD-Grafik gerechnet wurde. Ob es am Vulkan-Teil von llama.cpp liegt, am Grafiktreiber oder an der Grafik selbst, haben wir nicht mehr untersucht.
- **Parallele Anfragen** verschoben auch hier die Zahlen, auf beiden Rechnern und mit beiden Programmen. Auf dem Mini-PC änderte sich dabei auch der Text. Nur im deterministischen Modus von vLLM blieb alles gleich.

## Was hilft?

Es gibt drei Wege. Welcher passt, hängt davon ab, wie viel Kontrolle man über die Technik hat.

**1. Volle Kontrolle: ein frischer Prozess pro Anfrage.** Das ist der Weg, der in unseren Tests ausnahmslos funktioniert hat, mit allen drei Modellen und bei Qwen3-8B mit festem Seed sogar bei Temperatur 0,7. Dazu gehört:

- jede Anfrage in einem neu gestarteten Prozess, ohne Vorgeschichte
- keine gleichzeitigen Anfragen
- exakt dieselbe Software, dasselbe Modell, dieselbe Konfiguration

Der letzte Punkt ist wichtiger, als er klingt. Schon Flash Attention an oder aus verschiebt die Zahlen (Schritt 4), und die Zahl der Plätze auf dem Server ändert, welche Antwort herauskommt (Schritt 5). Bei einem Update der Software oder des Grafiktreibers ist dasselbe zu erwarten; getestet haben wir das nicht. Der Haken an diesem Weg: Das Modell muss für jede Anfrage neu geladen werden, und mehrere Nutzer gleichzeitig gehen nicht. Für einen Produktivbetrieb ist das kaum praktikabel, für die Nachprüfung einzelner Ergebnisse schon.

**2. Software, die dafür gebaut ist.** vLLM hat den deterministischen Modus aus der Gegenprobe (VLLM_BATCH_INVARIANT=1), SGLang einen vergleichbaren (`--enable-deterministic-inference`). Thinking Machines hat mit diesem Ansatz aus 80 verschiedenen Antworten eine gemacht: 1.000 von 1.000 identisch. Auch in unserer Gegenprobe hat der Modus gewirkt, kostet aber Zeit: bei uns das Doppelte bis Zwölffache.

Die Einschränkung: Der Modus ist laut vLLM noch im Beta-Stadium und läuft derzeit nur auf NVIDIA-GPUs ab der Ampere-Generation, auf AMDs Rechenzentrums-Beschleunigern (Instinct) und auf Intel-Beschleunigern mit Triton. Mein Mini-PC gehört nicht dazu, und der Modus von SGLang startet auf dieser Grafikeinheit laut einem offenen Fehlerbericht ebenfalls nicht.

**3. Cloud-APIs: keine Garantie.** Bei OpenAI gibt es einen Seed-Parameter, laut Dokumentation aber „Beta“, „best effort“ und ausdrücklich: „Determinism is not guaranteed.“ Anthropic geht den umgekehrten Weg: In der API-Version vom 21. Oktober 2026 führt die Angabe einer Temperatur zu einem Fehler, und laut Dokumentation ersetzt keine andere Einstellung sie. Neuere Claude-Modelle akzeptierten schon vorher nur den Standardwert. Wer eine Cloud-API nutzt, teilt sich die Server mit anderen (Schritt 5) und hat keinen Einfluss auf Konfiguration und Updates. Garantieren lässt sich Reproduzierbarkeit dort nicht.

**Und für das Audit vom Meetup?** Dort lautet meine Empfehlung: aufzeichnen statt nachrechnen. Prompt, Modell mit genauer Version, Parameter, Zeitpunkt und die vollständige Antwort speichern. Nachvollziehbar ist ein Ergebnis dann, weil man belegen kann, wie es entstanden ist. Nicht, weil man es auf Knopfdruck noch einmal erzeugen kann. Wer zusätzlich nachrechnen können will, braucht Weg 1 oder 2 und muss dafür Software und Hardware einfrieren.

## Fazit

Gleiche Frage, andere Antwort: Bei Temperatur null ist das kein Zufall und kein Fehler der Grafikkarte. Die Rechnung selbst ist deterministisch. Aber ihr Ergebnis hängt von mehr ab als vom Prompt: von anderen Anfragen im selben Stapel, von der Konfiguration des Servers und bei llama.cpp mit Vulkan sogar davon, was der Server vorher gerechnet hat. An knappen Stellen reichen winzige Rundungsunterschiede, die sich durch die Rechnung fortpflanzen, und das Modell schreibt eine andere Geschichte.

Temperatur null oder ein fester Seed schalten nur den gewollten Zufall ab. Für Reproduzierbarkeit reicht das nicht. Wer echte Reproduzierbarkeit braucht, muss die gesamte Kette kontrollieren. Alle anderen sollten nicht darauf bauen, dass dieselbe Frage dieselbe Antwort liefert, und lieber festhalten, welche Antwort sie bekommen haben.

Die Messskripte und alle Rohdaten liegen auf GitHub: https://github.com/ancroo/experiment-llm-determinism

Wer einen Fehler findet oder die Tests auf anderer Hardware nachmisst: Ich freue mich über Hinweise.

## Quellen

- Horace He / Thinking Machines Lab (2025): Defeating Nondeterminism in LLM Inference. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Jiayi Yuan et al. (2025): Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference. NeurIPS 2025, arXiv:2506.09501. https://arxiv.org/abs/2506.09501
- Ingonyama (2024): Solving Reproducibility Challenges in Deep Learning and LLMs: Our Journey. https://www.ingonyama.com/post/solving-reproducibility-challenges-in-deep-learning-and-llms-our-journey
- Sanjif Shanmugavelu et al. (2024): Impacts of floating-point non-associativity on reproducibility for HPC and deep learning applications. SC24-Workshops, IEEE. https://doi.org/10.1109/SCW63240.2024.00028
- Qwen3-8B Modellbeschreibung: https://huggingface.co/Qwen/Qwen3-8B
- gpt-oss-120b Modellbeschreibung: https://huggingface.co/openai/gpt-oss-120b
- vLLM, Batch Invariance: https://docs.vllm.ai/en/latest/features/batch_invariance.html
- SGLang, Deterministic Inference: https://docs.sglang.io/advanced_features/deterministic_inference.html
- SGLang Issue #48968 (Strix Halo): https://github.com/sgl-project/sglang/issues/48968
- OpenAI API Reference, Parameter seed: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/
- Anthropic, Migration zur API-Version 2026-10-21: https://platform.claude.com/docs/en/api/migration-2026-10-21
