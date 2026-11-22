Man stellt einem LLM zweimal dieselbe Frage, mit Temperatur null und festem Seed, und bekommt trotzdem zwei verschiedene Antworten.

Das erzählte mir auf einer KI-Veranstaltung jemand aus der Finanzbranche. Sein Interesse galt der Auditierbarkeit von LLMs. Sein Fazit: „LLMs können nicht deterministisch betrieben werden.“

Ich sah es anders: „Es ist eine Kette von Matrixmultiplikationen. Gleiche Eingabe, gleiches Ergebnis.“

Auf der Veranstaltung gab es keine Einigung.

Also haben Claude und ich nachgemessen. Auf eigener Hardware, mit offenen Modellen und in über 3.000 Messläufen.

Bei Temperatur 0,7 gab es bei 30 Anfragen 30 verschiedene Geschichten, so ist es gewollt. Bei Temperatur null waren 28 von 30 identisch, obwohl die Anfragen einzeln nacheinander kamen und sonst niemand auf dem Server war. Die beiden Abweichungen entstanden an Stellen, an denen zwei Wörter fast gleichauf lagen (z. B. 42,7 % zu 41,9 %), da reicht schon ein winziger Rundungsunterschied.

In unseren Tests hat die Grafikkarte nicht gewürfelt: Bei gleicher Ausgangslage kamen dieselben Zahlen heraus. Überrascht hat mich aber, was alles zur Ausgangslage gehört. Die Antwort hängt davon ab, was der Server (llama.cpp mit Vulkan auf AMD-Grafik) vorher gerechnet hat. Nach jedem Neustart kommt wieder exakt dieselbe Folge von Antworten. Der Server hat quasi ein Gedächtnis.

Danach haben wir das Ganze mit einem anderen, größeren Modell wiederholt (gpt-oss-120b). Gleiches Muster, nur stärker: 30 Anfragen, 26 verschiedene Antworten.

Dazu kommen gleichzeitige Anfragen anderer Nutzer. Sie verändern die Rechnung zusätzlich, und an knappen Stellen kann dann die Antwort kippen.

Im Endeffekt hatten wir also beide recht. Die Rechnung ist deterministisch, ihr Ergebnis hängt aber von mehr ab als vom eigenen Prompt. Temperatur null schaltet nur den gewollten „Zufall“ ab.

In unseren Tests war ein frisch gestarteter Prozess pro Anfrage ausnahmslos reproduzierbar. Alternativ hat die Server-Software vLLM einen deterministischen Modus. Der hat bei uns funktioniert, brauchte auf einer NVIDIA RTX 4070 aber pro Anfrage doppelt bis zwölfmal so lange, je nach Modell.

Das Gedächtnis des Servers gab es bei vLLM übrigens auch ohne diesen Modus nicht, und in einem Nachtest auf derselben Grafikkarte auch bei llama.cpp nicht. Bei uns trat es nur auf, wenn über Vulkan auf der AMD-Grafik gerechnet wurde. Woran genau es liegt, haben wir nicht mehr untersucht.

Für die Auditierbarkeit ist Determinismus vermutlich ohnehin die falsche Frage. Nachvollziehbar wird ein Ergebnis, wenn man es aufzeichnet: Prompt, Modellversion, Parameter und Antwort. Nachrechnen muss man es dafür nicht. Und nur weil eine Antwort reproduzierbar ist, ist sie noch lange nicht richtig. Entscheidend ist, ob sie brauchbar ist.

Das ist aber ein anderes Experiment ;-)

Der ganze Versuch mit allen Zahlen und Quellen steht im Artikel: https://lnkd.in/p/e-ZDXrbf
Messskripte und Rohdaten (github): https://lnkd.in/e5K7nj_v
