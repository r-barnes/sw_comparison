<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="ru">
<context>
    <name>EMBLGenbankAbstractDocument</name>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="57"/>
        <source>The file contains features of another remote GenBank file. These features have been skipped.</source>
        <translation>Файл содержит аннотации другого GenBank файла. Эти аннотации проигнорированы.</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="58"/>
        <source>The file contains joined annotations with regions, located on different strands. All such joined parts will be stored on the same strand.</source>
        <translation>Файл содержит аннотации присоединенные к регионам, расположенным на другой цепи. Все такие аннотации будут сохранены на одной цепи.</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="59"/>
        <source>Location parsing error.</source>
        <translation>Location parsing error.</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="60"/>
        <source>The number of valid sequence characters does not coincide with the declared size in the sequence header.</source>
        <translation>Число допустимых символов последовательности не совпадает с заявленным размером в заголовке последовательности.</translation>
    </message>
</context>
<context>
    <name>LocationParser</name>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="625"/>
        <source>Ignoring remote entry</source>
        <translation>Ignoring remote entry</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="626"/>
        <source>Ignoring different strands in JOIN</source>
        <translation>Ignoring different strands in JOIN</translation>
    </message>
</context>
<context>
    <name>QObject</name>
    <message>
        <location filename="../src/BAMUtils.cpp" line="97"/>
        <source>Fail to open &quot;%1&quot; for reading</source>
        <translation>Невозможно открыть &quot;%1&quot; для чтения</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="101"/>
        <source>Fail to read the header from the file: &quot;%1&quot;</source>
        <translation>Невозможно прочитать заголовок файла: &quot;%1&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="105"/>
        <source>Can not build the fasta index for the file: &quot;%1&quot;</source>
        <translation>Невозможно построить индекс для файла: &quot;%1&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="109"/>
        <source>Error parsing the reads from the file: &quot;%1&quot;</source>
        <translation>Ошибка чтения ридов из файла: &quot;%1&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="113"/>
        <source>Truncated file: &quot;%1&quot;</source>
        <translation>Обрезанный файл: &quot;%1&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="783"/>
        <source>Can&apos;t open file with given url: %1.</source>
        <translation>Can&apos;t open file with given url: %1.</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="484"/>
        <source>Invalid database user permissions set, so UGENE unable to use this database. Connect to your system administrator to fix the issue.</source>
        <translation>Установлены неверные права пользователя базы данных, таким образом UGENE не может использовать эту базу данных. Свяжитесь с вашим администратором для устранения проблемы.</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteBlobInputStream.cpp" line="96"/>
        <source>Can not read data. The database is closed or the data were changed.</source>
        <translation>Невозможно прочитать данные. База дынных закрыта или данные были изменены.</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteBlobOutputStream.cpp" line="49"/>
        <source>Can not write data. The database is closed or the data were changed.</source>
        <translation>Невозможно записать данные. База данных закрыта или данные были изменены.</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="787"/>
        <source>Unexpected number of modified objects. Expected: 1, actual: %1</source>
        <translation>Unexpected number of modified objects. Expected: 1, actual: %1</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="271"/>
        <location filename="../src/FastaFormat.cpp" line="313"/>
        <location filename="../src/FastqFormat.cpp" line="415"/>
        <source>Document sequences were merged</source>
        <translation>Последовательности были соединены</translation>
    </message>
    <message>
        <location filename="../src/PDWFormat.cpp" line="166"/>
        <location filename="../src/StockholmFormat.cpp" line="741"/>
        <source>The document is not created by UGENE</source>
        <translation>Документ создан не в UGENE</translation>
    </message>
</context>
<context>
    <name>U2::ABIFormat</name>
    <message>
        <location filename="../src/ABIFormat.cpp" line="54"/>
        <source>ABIF</source>
        <translation>ABIF</translation>
    </message>
    <message>
        <location filename="../src/ABIFormat.cpp" line="55"/>
        <source>A chromatogram file format</source>
        <translation>Формат типа хромотограмма</translation>
    </message>
    <message>
        <location filename="../src/ABIFormat.cpp" line="94"/>
        <source>Not a valid ABIF file: %1</source>
        <translation>Некорректный ABIF файл: %1</translation>
    </message>
    <message>
        <location filename="../src/ABIFormat.cpp" line="121"/>
        <source>Failed to load sequence from ABI file %1</source>
        <translation>Невозможно загрузить последоватлеьность из ABI файла %1</translation>
    </message>
    <message>
        <location filename="../src/ABIFormat.cpp" line="452"/>
        <source>Undefined sequence alphabet</source>
        <translation>Неизвестный алфавит</translation>
    </message>
</context>
<context>
    <name>U2::ACEFormat</name>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="51"/>
        <source>ACE</source>
        <translation>ACE</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="214"/>
        <source>Line is too long</source>
        <translation>Строка слишком длинная</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="350"/>
        <source>A name is not match with AF names</source>
        <translation>Имя не совпадает с именами AF</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="389"/>
        <source>First line is not an ace header</source>
        <translation>Первая строка не является заголовком ACE</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="392"/>
        <source>No contig count tag in the header line</source>
        <translation>Отсутствует тег числа контигов в строке заголовка</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="405"/>
        <source>Must be CO keyword</source>
        <translation>Ожидается ключевое слово CO</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="213"/>
        <source>Unexpected end of file</source>
        <translation>Неожиданный конец файла</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="415"/>
        <source>There is no note about reads count</source>
        <translation>Отсуствтует информация о числе считываний</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="224"/>
        <location filename="../src/ace/AceFormat.cpp" line="267"/>
        <source>There is no AF note</source>
        <translation>Отсутствует</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="52"/>
        <source>ACE is a format used for storing information about genomic confgurations</source>
        <translation>ACE это формат используемый для хранения информации о геномных конфигурациях</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="225"/>
        <location filename="../src/ace/AceFormat.cpp" line="282"/>
        <source>A name is duplicated</source>
        <translation>Повторные вхождения имени</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="232"/>
        <source>No consensus</source>
        <translation>Отсутствует консенсус</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="244"/>
        <source>BQ keyword hasn&apos;t been found</source>
        <translation>Не найдено ключевое слово BQ</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="247"/>
        <source>Bad consensus data</source>
        <translation>Плохие данные в консенсусе</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="274"/>
        <location filename="../src/ace/AceFormat.cpp" line="279"/>
        <source>Bad AF note</source>
        <translation>Плохое примечание AF</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="309"/>
        <source>There is no read note</source>
        <translation>Отсутствует</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="317"/>
        <source>No sequence</source>
        <translation>Нет последовательности</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="329"/>
        <source>QA keyword hasn&apos;t been found</source>
        <translation>Ключевое слово QA не было найдено</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="335"/>
        <location filename="../src/ace/AceFormat.cpp" line="338"/>
        <source>QA error no clear range</source>
        <translation>Ошибка QA нет четкого региона</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="342"/>
        <source>QA error bad range</source>
        <translation>Ошибка QA: плохой регион</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="347"/>
        <source>Bad sequence data</source>
        <translation>Некорректные данные в последовательности</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="463"/>
        <source>Alphabet unknown</source>
        <translation>Неизвестный алфавит</translation>
    </message>
    <message>
        <location filename="../src/ace/AceFormat.cpp" line="489"/>
        <source>File doesn&apos;t contain any msa objects</source>
        <translation>Файл не содержит msa объектов</translation>
    </message>
</context>
<context>
    <name>U2::ASNFormat</name>
    <message>
        <location filename="../src/ASNFormat.cpp" line="51"/>
        <source>MMDB</source>
        <translation>MMDB</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.h" line="148"/>
        <source>read error occurred</source>
        <translation>Ошибка чтения</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.h" line="160"/>
        <source>biostruct3d obj loading error: %1</source>
        <translation>Ошибка загрузки трехмерной структуры: %1</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.cpp" line="52"/>
        <source>ASN is a format used my the Molecular Modeling Database (MMDB)</source>
        <translation>ASN это формат используемый Molecular Modeling Database (MMDB)</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.cpp" line="197"/>
        <location filename="../src/ASNFormat.cpp" line="570"/>
        <source>Unknown error occurred</source>
        <translation>Неизвестная ошибка</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.cpp" line="556"/>
        <source>no root element</source>
        <translation>Отсутствует корневой элемент</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.cpp" line="562"/>
        <source>states stack is not empty</source>
        <translation>Стек состояний не пуст</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.cpp" line="585"/>
        <source>First line is too long</source>
        <translation>Первая строка слишком длинная</translation>
    </message>
    <message>
        <location filename="../src/ASNFormat.cpp" line="73"/>
        <source>Standard residue dictionary not found</source>
        <translation>Стандартный словарь остатков не найден</translation>
    </message>
</context>
<context>
    <name>U2::AbstractVariationFormat</name>
    <message>
        <location filename="../src/AbstractVariationFormat.cpp" line="64"/>
        <source>SNP formats are used to store single-nucleotide polymorphism data</source>
        <translation>SNP форматы используются для сохранения полиморфизма однонуклеотидных данных</translation>
    </message>
    <message>
        <location filename="../src/AbstractVariationFormat.cpp" line="140"/>
        <source>Line %1: There are too few columns in this line. The line was skipped.</source>
        <translation>Строка %1: Слишком мало столбцов в этой строке. Строка была проигнорирована.</translation>
    </message>
</context>
<context>
    <name>U2::AceImporter</name>
    <message>
        <location filename="../src/ace/AceImporter.cpp" line="154"/>
        <source>ACE file importer</source>
        <translation>Импорт ACE файла</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImporter.cpp" line="158"/>
        <source>ACE files importer is used to convert conventional ACE files into UGENE database format.Having ACE file converted into UGENE DB format you get an fast and efficient interface to your data with an option to change the content</source>
        <translation>Импорт ACE файлов используется для преобразования обычных ACE файлов в формат базы данных UGENE. Преобразовав ACE файл в формат UGENE DB вы получите быстрый и эффективный интерфейс к вашим данным с возможностью изменять содержимое файла</translation>
    </message>
</context>
<context>
    <name>U2::AceImporterTask</name>
    <message>
        <location filename="../src/ace/AceImporter.cpp" line="55"/>
        <source>ACE file import: %1</source>
        <translation>Импорт ACE файла: %1</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImporter.cpp" line="68"/>
        <source>Dbi ref is invalid</source>
        <translation>Dbi ref is invalid</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImporter.cpp" line="84"/>
        <source>Can&apos;t create a temporary database</source>
        <translation>Невозможно создать временную базу данных</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImporter.cpp" line="141"/>
        <source>Failed to get load task for : %1</source>
        <translation>Не удалось получить загрузку задачи для: %1</translation>
    </message>
</context>
<context>
    <name>U2::AprFormat</name>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="61"/>
        <source>Unexpected end of file</source>
        <translation>Неожиданный конец файла</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="71"/>
        <source>There is no sequences in alignment</source>
        <translation>Выравнивание не содержит последовательностей</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="98"/>
        <source>Attempt to find any number in the string failed</source>
        <translation>Attempt to find any number in the string failed</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="150"/>
        <source>Vector NTI/AlignX</source>
        <translation>Vector NTI/AlignX</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="151"/>
        <source>Vector NTI/AlignX is a Vector NTI format for multiple alignment</source>
        <translation>Vector NTI/AlignX это Vector NTI формат для множественных выравниваний</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="172"/>
        <source>Open in read-only mode</source>
        <translation>Открыть только для чтения</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="182"/>
        <source>File doesn&apos;t contain any msa objects</source>
        <translation>Файл не содержит msa объектов</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="206"/>
        <source>Illegal header line</source>
        <translation>Неправильная строка заголовка</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="221"/>
        <source>Sequences not found</source>
        <translation>Последовательности не найдены</translation>
    </message>
    <message>
        <location filename="../src/apr/AprFormat.cpp" line="227"/>
        <source>Alphabet is unknown</source>
        <translation>Неизвестный алфавит</translation>
    </message>
</context>
<context>
    <name>U2::AprImporter</name>
    <message>
        <location filename="../src/apr/AprImporter.cpp" line="116"/>
        <source>Vector NTI/AlignX file importer</source>
        <translation>Инструмент для импорта Vector NTI/AlignX файлов</translation>
    </message>
    <message>
        <location filename="../src/apr/AprImporter.cpp" line="120"/>
        <source>Vector NTI/AlignX files importer is used to convert conventional APR files to a multiple sequence alignment formats</source>
        <translation>Инструмент для импорта Vector NTI/AlignX файлов используется для конвертации APR файлов в формат множественного выравнивания</translation>
    </message>
    <message>
        <location filename="../src/apr/AprImporter.cpp" line="135"/>
        <source>Convert to another format:</source>
        <translation>Конвертировать в другой формат:</translation>
    </message>
</context>
<context>
    <name>U2::AprImporterTask</name>
    <message>
        <location filename="../src/apr/AprImporter.cpp" line="54"/>
        <source>APR file import: %1</source>
        <translation>APR file import: %1</translation>
    </message>
    <message>
        <location filename="../src/apr/AprImporter.cpp" line="65"/>
        <location filename="../src/apr/AprImporter.cpp" line="91"/>
        <source>Empty destination url</source>
        <translation>Empty destination url</translation>
    </message>
    <message>
        <location filename="../src/apr/AprImporter.cpp" line="71"/>
        <source>Invalid I/O environment!</source>
        <translation>Invalid I/O environment!</translation>
    </message>
</context>
<context>
    <name>U2::BAMUtils</name>
    <message>
        <location filename="../src/BAMUtils.cpp" line="180"/>
        <source>There is no header in the SAM file &quot;%1&quot;. The header information will be generated automatically.</source>
        <translation>Нет заголовка в SAM файле &quot;%1&quot;. Необходимая информация будет сгенерирована автоматически.</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="339"/>
        <source>Sort bam file: &quot;%1&quot; using %2 Mb of memory. Result sorted file is: &quot;%3&quot;</source>
        <translation>Сортировка bam файла: &quot;%1&quot; использует %2 Mb памяти. Сортированный файл: &quot;%3&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="352"/>
        <source>Merging BAM files: &quot;%1&quot;. Resulting merged file is: &quot;%2&quot;</source>
        <translation>Слияние BAM файлов: &quot;%1&quot;. Соединенный файл: &quot;%2&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="369"/>
        <source>Remove PCR duplicate in BAM file: &quot;%1&quot;. Resulting  file is: &quot;%2&quot;</source>
        <translation>Удаление PCR повторов в BAM файле: &quot;%1&quot;. Результирующий файл: &quot;%2&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="457"/>
        <source>Build index for bam file: &quot;%1&quot;</source>
        <translation>Построение индекса для bam файла: &quot;%1&quot;</translation>
    </message>
    <message>
        <location filename="../src/BAMUtils.cpp" line="724"/>
        <source>Wrong line in a SAM file.</source>
        <translation>Wrong line in a SAM file.</translation>
    </message>
</context>
<context>
    <name>U2::BedFormat</name>
    <message>
        <location filename="../src/BedFormat.cpp" line="100"/>
        <source>The BED (Browser Extensible Data) format was developed by UCSC for displaying transcript structures in the genome browser.</source>
        <translation>BED (Browser Extensible Data) формат был разработан UCSC для отображения транскриптных структур в геномном браузере.</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="136"/>
        <source>File &quot;%1&quot; contains too many annotation tables to be displayed. However, you can process these data using pipelines built with Workflow Designer.</source>
        <translation>File &quot;%1&quot; contains too many annotation tables to be displayed. However, you can process these data using pipelines built with Workflow Designer.</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="863"/>
        <source>BED parsing error: incorrect format of the &apos;track&apos; header line!</source>
        <translation>BED parsing error: incorrect format of the &apos;track&apos; header line!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="655"/>
        <source>BED parsing error: unexpected number of fields in the first annotations line!</source>
        <translation>BED parsing error: unexpected number of fields in the first annotations line!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="99"/>
        <source>BED</source>
        <translation>BED</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="696"/>
        <source>The file does not contain valid annotations!</source>
        <translation>Файл не содержит корректных аннотаций!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="932"/>
        <source>BED parsing error: incorrect number of fields at line %1!</source>
        <translation>BED parsing error: incorrect number of fields at line %1!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="936"/>
        <source>BED parsing error: a field at line %1 is empty!</source>
        <translation>BED parsing error: a field at line %1 is empty!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="940"/>
        <source>BED parsing error: incorrect coordinates at line %1!</source>
        <translation>BED parsing error: incorrect coordinates at line %1!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="944"/>
        <source>BED parsing error: incorrect score value &apos;%1&apos; at line %2!</source>
        <translation>BED parsing error: incorrect score value &apos;%1&apos; at line %2!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="951"/>
        <source>BED parsing error: incorrect strand value &apos;%1&apos; at line %2!</source>
        <translation>BED parsing error: incorrect strand value &apos;%1&apos; at line %2!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="958"/>
        <source>BED parsing error: incorrect thick coordinates at line %1!</source>
        <translation>BED parsing error: incorrect thick coordinates at line %1!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="962"/>
        <source>BED parsing error: incorrect itemRgb value &apos;%1&apos; at line %2!</source>
        <translation>BED parsing error: incorrect itemRgb value &apos;%1&apos; at line %2!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="969"/>
        <source>BED parsing error: incorrect value of the block parameters at line %1!</source>
        <translation>BED parsing error: incorrect value of the block parameters at line %1!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="403"/>
        <source>Starting BED saving: &apos;%1&apos;</source>
        <translation>Сохранение BED: &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="414"/>
        <source>Can not convert GObject to AnnotationTableObject</source>
        <translation>Can not convert GObject to AnnotationTableObject</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="428"/>
        <source>Can not detect chromosome name. &apos;Chr&apos; name will be used.</source>
        <translation>Can not detect chromosome name. &apos;Chr&apos; name will be used.</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="444"/>
        <source>You are trying to save joined annotation to BED format! The joining will be lost</source>
        <translation>Вы пытаетесь сохранить связанные аннотации в BED формат! Соединения будут потеряны</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="507"/>
        <source>BED saving error: incorrect thick coordinates in the first annotation!</source>
        <translation>BED saving error: incorrect thick coordinates in the first annotation!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="521"/>
        <source>BED saving error: incorrect block fields in the first annotation!</source>
        <translation>BED saving error: incorrect block fields in the first annotation!</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="529"/>
        <source>BED saving: detected %1 fields per line for file &apos;%2&apos;</source>
        <translation>BED saving: detected %1 fields per line for file &apos;%2&apos;</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="538"/>
        <source>BED saving error: an annotation is expected to have &apos;%1&apos; qualifier, but it is absent! Skipping the annotation.</source>
        <translation>BED saving error: an annotation is expected to have &apos;%1&apos; qualifier, but it is absent! Skipping the annotation.</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="592"/>
        <source>BED saving error: an annotation is expected to have the block qualifiers! Skipping the annotation.</source>
        <translation>BED saving error: an annotation is expected to have the block qualifiers! Skipping the annotation.</translation>
    </message>
    <message>
        <location filename="../src/BedFormat.cpp" line="614"/>
        <source>Finished BED saving: &apos;%1&apos;</source>
        <translation>Сохранение BED закончено: &apos;%1&apos;</translation>
    </message>
</context>
<context>
    <name>U2::BgzipTask</name>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="49"/>
        <source>Bgzip Compression task</source>
        <translation>Bgzip Compression task</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="55"/>
        <source>Start bgzip compression &apos;%1&apos;</source>
        <translation>Начало сжатия bgzip &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="57"/>
        <source>IOAdapterRegistry is NULL!</source>
        <translation>IOAdapterRegistry is NULL!</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="60"/>
        <source>IOAdapterFactory is NULL!</source>
        <translation>IOAdapterFactory is NULL!</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="62"/>
        <source>Can not create IOAdapter!</source>
        <translation>Can not create IOAdapter!</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="66"/>
        <source>Can not open input file &apos;%1&apos;</source>
        <translation>Can not open input file &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="76"/>
        <source>Can not open output file &apos;%2&apos;</source>
        <translation>Невозможно открыть выходной файл &apos;%2&apos;</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="90"/>
        <source>Error reading file</source>
        <translation>Ошибка чтения файла</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="95"/>
        <source>Error writing to file</source>
        <translation>Ошибка записи в файл</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="102"/>
        <source>Bgzip compression finished</source>
        <translation>Сжатие bgzip завершено</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="107"/>
        <source>Bgzip compression task was finished with an error: %1</source>
        <translation>Сжатие bgzip закончилось с ошибкой: %1</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="109"/>
        <source>Bgzip compression task was finished. A new bgzf file is: &lt;a href=&quot;%1&quot;&gt;%2&lt;/a&gt;</source>
        <translation>Сжатие bgzip завершено. Новый bgzf файл: &lt;a href=&quot;%1&quot;&gt;%2&lt;/a&gt;</translation>
    </message>
</context>
<context>
    <name>U2::CalculateSequencesNumberTask</name>
    <message>
        <location filename="../src/tasks/CalculateSequencesNumberTask.cpp" line="29"/>
        <source>Calculate sequences number</source>
        <translation>Calculate sequences number</translation>
    </message>
</context>
<context>
    <name>U2::CloneAssemblyWithReferenceToDbiTask</name>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="43"/>
        <source>Clone assembly object to the destination database</source>
        <translation>Клонирование объекта сборки в базу данных</translation>
    </message>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="51"/>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="52"/>
        <source>Invalid assembly ID</source>
        <translation>Invalid assembly ID</translation>
    </message>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="53"/>
        <source>Invalid source database reference</source>
        <translation>Invalid source database reference</translation>
    </message>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="54"/>
        <source>Invalid destination database reference</source>
        <translation>Invalid destination database reference</translation>
    </message>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="72"/>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="78"/>
        <source>Can&apos;t get the cloned object</source>
        <translation>Can&apos;t get the cloned object</translation>
    </message>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="74"/>
        <source>Unexpected result object: expect AssemblyObject, got %1 object</source>
        <translation>Unexpected result object: expect AssemblyObject, got %1 object</translation>
    </message>
    <message>
        <location filename="../src/ace/CloneAssemblyWithReferenceToDbiTask.cpp" line="80"/>
        <source>Unexpected result object: expect U2SequenceObject, got %1 object</source>
        <translation>Unexpected result object: expect U2SequenceObject, got %1 object</translation>
    </message>
</context>
<context>
    <name>U2::ClustalWAlnFormat</name>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="65"/>
        <source>Clustalw is a format for storing multiple sequence alignments</source>
        <translation>Clustalw это формат для сохранения множественных выравниваний</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="90"/>
        <source>Illegal header line</source>
        <translation>Неправильная строка заголовка</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="111"/>
        <source>Error parsing file</source>
        <translation>Ошибка разбора файла</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="123"/>
        <source>Invalid alignment format</source>
        <translation>Неверный формат выравнивания</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="161"/>
        <source>Incorrect number of sequences in block</source>
        <translation>Неверное количество последовательностей в блоке</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="167"/>
        <source>Sequence names are not matched</source>
        <translation>Встретилось непарное имя последовательности</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="191"/>
        <source>Alphabet is unknown</source>
        <translation>Неизвестный алфавит</translation>
    </message>
    <message>
        <location filename="../src/ClustalWAlnFormat.cpp" line="64"/>
        <source>CLUSTALW</source>
        <translation>CLUSTALW</translation>
    </message>
</context>
<context>
    <name>U2::ConvertAceToSqliteTask</name>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="47"/>
        <source>Convert ACE to UGENE database (%1)</source>
        <translation>Преобразование ACE в UGENE database формат(%1)</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="58"/>
        <source>Converting assembly from %1 to %2 started</source>
        <translation>Ковертация сборки из %1 в %2 начата</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="66"/>
        <source>IOAdapterFactory is NULL</source>
        <translation>IOAdapterFactory is NULL</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="70"/>
        <source>Can&apos;t open file &apos;%1&apos;</source>
        <translation>Невозможно открыть файл &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="83"/>
        <source>DBI is NULL</source>
        <translation>DBI is NULL</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="85"/>
        <source>Object DBI is NULL</source>
        <translation>Object DBI is NULL</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="88"/>
        <source>Importing</source>
        <translation>Импорт</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="213"/>
        <source>Assembly DBI is NULL</source>
        <translation>Assembly DBI is NULL</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="136"/>
        <source>Sequence DBI is NULL</source>
        <translation>Sequence DBI is NULL</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="155"/>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="198"/>
        <source>Invalid source file</source>
        <translation>Invalid source file</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="219"/>
        <source>Packing reads for assembly &apos;%1&apos; (%2 of %3)</source>
        <translation>Упаковка ридов для сборки &apos;%1&apos; (%2 от %3)</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="245"/>
        <source>Attribute DBI is NULL</source>
        <translation>Attribute DBI is NULL</translation>
    </message>
    <message>
        <location filename="../src/ace/ConvertAceToSqliteTask.cpp" line="274"/>
        <source>Warning: incorrect maxProw == %1, probably packing was not done! Attribute was not set</source>
        <translation>Warning: incorrect maxProw == %1, probably packing was not done! Attribute was not set</translation>
    </message>
</context>
<context>
    <name>U2::ConvertAssemblyToSamTask</name>
    <message>
        <location filename="../src/tasks/ConvertAssemblyToSamTask.cpp" line="89"/>
        <source>Given file is not valid UGENE database file</source>
        <translation>Данный файл не является корректным файлом формата базы данных UGENE</translation>
    </message>
</context>
<context>
    <name>U2::ConvertSnpeffVariationsToAnnotationsTask</name>
    <message>
        <location filename="../src/tasks/ConvertSnpeffVariationsToAnnotationsTask.cpp" line="50"/>
        <source>Convert SnpEff variations to annotations task</source>
        <translation>Convert SnpEff variations to annotations task</translation>
    </message>
</context>
<context>
    <name>U2::DNAQualityIOUtils</name>
    <message>
        <location filename="../src/DNAQualityIOUtils.cpp" line="61"/>
        <source>No IO adapter found for URL: %1</source>
        <translation>No IO adapter found for URL: %1</translation>
    </message>
</context>
<context>
    <name>U2::Database</name>
    <message>
        <location filename="../src/Database.cpp" line="46"/>
        <source>Not a valid S3-database file: %1</source>
        <translation>Not a valid S3-database file: %1</translation>
    </message>
    <message>
        <location filename="../src/Database.cpp" line="56"/>
        <source>File already exists: %1</source>
        <translation>Файл уже существует: %1</translation>
    </message>
</context>
<context>
    <name>U2::DefaultConvertFileTask</name>
    <message>
        <location filename="../src/tasks/ConvertFileTask.cpp" line="125"/>
        <source>The formats are not compatible: %1 and %2</source>
        <translation>Форматы не совместимы: %1 и %2</translation>
    </message>
</context>
<context>
    <name>U2::DifferentialFormat</name>
    <message>
        <location filename="../src/DifferentialFormat.cpp" line="43"/>
        <source>Differential</source>
        <translation>Дифференциальный</translation>
    </message>
    <message>
        <location filename="../src/DifferentialFormat.cpp" line="45"/>
        <source>Differential format is a text-based format for representing Cuffdiff differential output files: expression, splicing, promoters and cds.</source>
        <translation>Дифференциальный формат это текстовый формат для представления выходных файлов Cuffdiff.</translation>
    </message>
    <message>
        <location filename="../src/DifferentialFormat.cpp" line="279"/>
        <source>Required column is missed: %1</source>
        <translation>Пропущен необходимый столбец: %1</translation>
    </message>
</context>
<context>
    <name>U2::Document</name>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="265"/>
        <location filename="../src/FastaFormat.cpp" line="289"/>
        <location filename="../src/FastqFormat.cpp" line="401"/>
        <location filename="../src/PDWFormat.cpp" line="153"/>
        <source>Document is empty.</source>
        <translation>Документ не содержит данных.</translation>
    </message>
</context>
<context>
    <name>U2::DocumentFormat</name>
    <message>
        <location filename="../src/DatabaseConnectionFormat.cpp" line="42"/>
        <source>Database connection</source>
        <translation>Подключение к базе данных</translation>
    </message>
    <message>
        <location filename="../src/DatabaseConnectionFormat.cpp" line="43"/>
        <source>A fake format that was added to implement shared database connection within existing document model.</source>
        <translation>Неверный формат, который был добавлен  в процессе реализации подключения к базе данных в рамках существующей модели документа.</translation>
    </message>
    <message>
        <location filename="../src/DatabaseConnectionFormat.cpp" line="77"/>
        <source>You have no permissions to modify the content of this database</source>
        <translation>У вас недостаточно прав чтобы изменять содержимое этой базы данных</translation>
    </message>
    <message>
        <location filename="../src/DatabaseConnectionFormat.cpp" line="96"/>
        <source>Empty object name</source>
        <translation>Имя объекта пусто</translation>
    </message>
</context>
<context>
    <name>U2::DocumentFormatUtils</name>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="136"/>
        <source>First line is not an ace header</source>
        <translation>Первая строка не является заголовком ACE</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="140"/>
        <source>There is no assemblies in input file</source>
        <translation>В файле нет сборок</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="156"/>
        <source>There are not enough assemblies</source>
        <translation>Недостаточно сборок</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="212"/>
        <location filename="../src/ace/AceImportUtils.cpp" line="464"/>
        <source>Unexpected end of file</source>
        <translation>Неожиданный конец файла</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="213"/>
        <source>Line is too long</source>
        <translation>Строка слишком длинная</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="220"/>
        <source>No contig count tag in the header line</source>
        <translation>Отсутствует тег числа контигов в строке заголовка</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="232"/>
        <source>Not enough parameters in current line</source>
        <translation>Недостаточно параметров в текущей строке</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="243"/>
        <source>Parameter is not a digit</source>
        <translation>Параметр не является цифрой</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="251"/>
        <source>There is no note about reads count</source>
        <translation>Отсуствтует информация о числе считываний</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="263"/>
        <source>A name is duplicated</source>
        <translation>Повторные вхождения имени</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="270"/>
        <source>No consensus</source>
        <translation>Отсутствует консенсус</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="280"/>
        <source>BQ keyword hasn&apos;t been found</source>
        <translation>Не найдено ключевое слово BQ</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="283"/>
        <source>Unexpected symbols in consensus data</source>
        <translation>Неоижданные символы в консенсусе</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="294"/>
        <source>Can&apos;t find a sequence name in current line</source>
        <translation>Невозможно найти имя последовательности в текущей строке</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="304"/>
        <source>An empty sequence name</source>
        <translation>Не указано имя последовательности</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="354"/>
        <source>Invalid AF tag</source>
        <translation>Invalid AF tag</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="371"/>
        <source>A name is duplicated: %1</source>
        <translation>Повторные вхождения имени: %1</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="377"/>
        <source>Not all reads were found</source>
        <translation>Не все риды были найдены</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="386"/>
        <location filename="../src/ace/AceImportUtils.cpp" line="390"/>
        <location filename="../src/ace/AceImportUtils.cpp" line="422"/>
        <location filename="../src/ace/AceImportUtils.cpp" line="434"/>
        <source>Bad AF note</source>
        <translation>Плохое примечание AF</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="460"/>
        <source>There is no read note</source>
        <translation>Отсутствует</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="473"/>
        <source>Invalid RD part</source>
        <translation>Неверная часть RD</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="474"/>
        <source>Can&apos;t find the RD tag</source>
        <translation>Can&apos;t find the RD tag</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="483"/>
        <source>QA keyword hasn&apos;t been found</source>
        <translation>Ключевое слово QA не было найдено</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="492"/>
        <source>QA error bad range</source>
        <translation>Ошибка QA: плохой регион</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="495"/>
        <source>Unexpected symbols in sequence data</source>
        <translation>Неоижданные символы в последовательности</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="497"/>
        <source>A name is not match with AF names</source>
        <translation>Имя не совпадает с именами AF</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="504"/>
        <source>Can&apos;t find clear range start in current line</source>
        <translation>Невозможно найти начало диапазона в текущей строке</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="505"/>
        <source>Clear range start is invalid</source>
        <translation>Неверное начало диапазона</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="512"/>
        <source>Can&apos;t find clear range end in current line</source>
        <translation>Невозможно найти конец диапазона в текущей строке</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="513"/>
        <source>Clear range end is invalid</source>
        <translation>Неверный конец диапазона</translation>
    </message>
    <message>
        <location filename="../src/ace/AceImportUtils.cpp" line="537"/>
        <source>There is no next element</source>
        <translation>Отсутствует следующий элемент</translation>
    </message>
    <message>
        <location filename="../src/tasks/ConvertFileTask.cpp" line="56"/>
        <source>Conversion file from %1 to %2</source>
        <translation>Преобразование файла %1 в %2</translation>
    </message>
    <message>
        <location filename="../src/tasks/MergeBamTask.cpp" line="45"/>
        <source>Merge BAM files with SAMTools merge</source>
        <translation>Слияние BAM файлов с SAMTools</translation>
    </message>
</context>
<context>
    <name>U2::EMBLGenbankAbstractDocument</name>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="575"/>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="380"/>
        <source>Annotation name is empty</source>
        <translation>Не указано имя аннотации</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="586"/>
        <source>Error parsing location</source>
        <translation>Не указан регион аннотации</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="682"/>
        <source>Error parsing sequence: unexpected empty line</source>
        <translation>Ошибка чтения последовательности: пустая строка</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="614"/>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="424"/>
        <source>Unexpected line format</source>
        <translation>Слишком длинная строка или неожиданный конец файла</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="127"/>
        <source>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</source>
        <translation>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="144"/>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="314"/>
        <source>Reading entry header</source>
        <translation>Чтение заголовка</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="222"/>
        <source>Merge error: found annotations without sequence</source>
        <translation>Ошибка слияния: обнаружена таблица аннотаций без соотв последовательности</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="470"/>
        <source>The file contains an incorrect data that describes a qualifier value. </source>
        <translation>Файл содержит некорректные данные, которые описываются value. </translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="471"/>
        <source>The value cannot contain a single quote character. The qualifier is &apos;%1&apos;</source>
        <translation>Значение не может содержать одиночные кавычки. Квалификатор &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="672"/>
        <source>Reading sequence %1</source>
        <translation>Чтение последовательности: %1</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="697"/>
        <source>Sequence is truncated</source>
        <translation>Последовательность повреждена</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="703"/>
        <source>Reading annotations %1</source>
        <translation>Чтение аннотаций: %1</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="713"/>
        <source>Invalid format of feature table</source>
        <translation>Таблица аннотаций повреждена</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="810"/>
        <source>Line is too long.</source>
        <translation>Слишком длинная строка.</translation>
    </message>
    <message>
        <location filename="../src/EMBLGenbankAbstractDocument.cpp" line="812"/>
        <source>IO error.</source>
        <translation>Ошибка чтения.</translation>
    </message>
    <message>
        <location filename="../src/EMBLPlainTextFormat.cpp" line="228"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="250"/>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="229"/>
        <source>Record is truncated.</source>
        <translation>Данные повреждены.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="387"/>
        <source>Annotation start position is empty</source>
        <translation>Не указано начало аннотации</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="392"/>
        <source>Annotation end position is empty</source>
        <translation>Не указан конец аннотации</translation>
    </message>
</context>
<context>
    <name>U2::EMBLPlainTextFormat</name>
    <message>
        <location filename="../src/EMBLPlainTextFormat.cpp" line="43"/>
        <source>EMBL</source>
        <translation>EMBL</translation>
    </message>
    <message>
        <location filename="../src/EMBLPlainTextFormat.cpp" line="47"/>
        <source>EMBL Flat File Format is a rich format for storing sequences and associated annotations</source>
        <translation>EMBL Flat File Format это формат для хранения последовательностей и их аннотаций</translation>
    </message>
    <message>
        <location filename="../src/EMBLPlainTextFormat.cpp" line="84"/>
        <source>ID is not the first line</source>
        <translation>Строка идентификатора должна идти первой</translation>
    </message>
    <message>
        <location filename="../src/EMBLPlainTextFormat.cpp" line="91"/>
        <source>Error parsing ID line</source>
        <translation>Неверный заголовок</translation>
    </message>
</context>
<context>
    <name>U2::FastaFormat</name>
    <message>
        <location filename="../src/FastaFormat.cpp" line="68"/>
        <source>FASTA format is a text-based format for representing either nucleotide sequences or peptide sequences, in which base pairs or amino acids are represented using single-letter codes. The format also allows for sequence names and comments to precede the sequences.</source>
        <translation>Формат FASTA является текстовым форматом для представления нуклеотидных или пептидных последовательностей, в котором пары оснований или аминокислоты представлены с использованием одно-буквенных кодов. Формат также позволяет задавать имена и комментарии к последовательностям.</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="188"/>
        <location filename="../src/FastaFormat.cpp" line="425"/>
        <source>Line is too long</source>
        <translation>Слишком длинная строка</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="191"/>
        <location filename="../src/FastaFormat.cpp" line="427"/>
        <source>First line is not a FASTA header</source>
        <translation>Неправильный заголовок FASTA</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="253"/>
        <source>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</source>
        <translation>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="281"/>
        <source>Sequence #%1 is processed</source>
        <translation>Sequence #%1 is processed</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="285"/>
        <source>The file format is invalid.</source>
        <translation>The file format is invalid.</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="295"/>
        <source>Loaded sequences: %1. 
</source>
        <translation>Loaded sequences: %1. 
</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="296"/>
        <source>Skipped sequences: %1. 
</source>
        <translation>Skipped sequences: %1. 
</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="297"/>
        <source>The following sequences are empty: 
%1</source>
        <translation>The following sequences are empty: 
%1</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="493"/>
        <source>Unreferenced sequence in the beginning of patterns: %1</source>
        <translation>Неиспользуемая последовательность в начале образцов: %1</translation>
    </message>
    <message>
        <location filename="../src/FastaFormat.cpp" line="65"/>
        <source>FASTA</source>
        <translation>FASTA</translation>
    </message>
</context>
<context>
    <name>U2::FastqFormat</name>
    <message>
        <location filename="../src/FastqFormat.cpp" line="55"/>
        <source>FASTQ</source>
        <translation>FASTQ</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="56"/>
        <source>FASTQ format is a text-based format for storing both a biological sequence (usually nucleotide sequence) and its corresponding quality scores.         Both the sequence letter and quality score are encoded with a single ASCII character for brevity.         It was originally developed at the Wellcome Trust Sanger Institute to bundle a FASTA sequence and its quality data,         but has recently become the de facto standard for storing the output of high throughput sequencing instruments.</source>
        <translation>Формат FASTQ является текстовым форматом для хранения биологических последовательностей (обычно нуклеотидных) и соответствующих им показателей качества. Последовательность и показатель качества кодируются при помощи одного символа ASCII для краткости. Изначально он был разработан в Wellcome Trust Sanger Institute для связи последовательности в формате FASTA и их данных качества, но в последнее время стал стандартом  для хранения выходных данных инструментов секвенирования.</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="147"/>
        <location filename="../src/FastqFormat.cpp" line="154"/>
        <source>Error while trying to find sequence name start</source>
        <translation>Не удалось найти начало имени последовательности</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="174"/>
        <location filename="../src/FastqFormat.cpp" line="206"/>
        <source>Error while reading sequence</source>
        <translation>Ошибка чтения последовательности</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="339"/>
        <source>Sequence name differs from quality scores name: %1 and %2</source>
        <translation>Sequence name differs from quality scores name: %1 and %2</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="358"/>
        <source>Bad quality scores: inconsistent size.</source>
        <translation>Bad quality scores: inconsistent size.</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="373"/>
        <source>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</source>
        <translation>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="568"/>
        <source>Not a valid FASTQ file, sequence name differs from quality scores name</source>
        <translation>Некорректный FASTQ файл, имя последовательности отличается от имени показателей качества</translation>
    </message>
    <message>
        <location filename="../src/FastqFormat.cpp" line="576"/>
        <source>Not a valid FASTQ file. Bad quality scores: inconsistent size.</source>
        <translation>Некорректный FASTQ файл. Плохие показатели качества: несовместимый размер.</translation>
    </message>
</context>
<context>
    <name>U2::FpkmTrackingFormat</name>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="86"/>
        <source>FPKM Tracking Format</source>
        <translation>FPKM Tracking Format</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="87"/>
        <source>The FPKM (fragments per kilobase of exon model per million mapped fragments) Tracking Format is a native Cufflinks format to output estimated expression values.</source>
        <translation>FPKM (fragments per kilobase of exon model per million mapped fragments) Tracking Format это внутренний Cufflinks формат для выходных значений оценки выражений.</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="120"/>
        <source>Internal error: qualifier with name &apos;%1&apos; and &apos;%2&apos; can&apos;t be added</source>
        <translation>Internal error: qualifier with name &apos;%1&apos; and &apos;%2&apos; can&apos;t be added</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="151"/>
        <source>FPKM Tracking Format parsing error: incorrect number of fields at line %1!</source>
        <translation>FPKM Tracking Format parsing error: incorrect number of fields at line %1!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="156"/>
        <source>FPKM Tracking Format parsing error: a field at line %1 is empty!</source>
        <translation>FPKM Tracking Format parsing error: a field at line %1 is empty!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="161"/>
        <source>FPKM Tracking Format parsing error: incorrect coordinates at line %1!</source>
        <translation>FPKM Tracking Format parsing error: incorrect coordinates at line %1!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="173"/>
        <source>FPKM Tracking Format parsing error: different sequence names were detected in an input file. Sequence name &apos;%1&apos; is used.</source>
        <translation>FPKM Tracking Format parsing error: different sequence names were detected in an input file. Sequence name &apos;%1&apos; is used.</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="189"/>
        <source>FPKM Tracking Format parsing error: tracking ID value is empty at line %1!</source>
        <translation>FPKM Tracking Format parsing error: tracking ID value is empty at line %1!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="213"/>
        <source>FPKM Tracking Format parsing error: incorrect length value at line %1!</source>
        <translation>FPKM Tracking Format parsing error: incorrect length value at line %1!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="219"/>
        <source>FPKM Tracking Format parsing error: incorrect coverage value at line %1!</source>
        <translation>FPKM Tracking Format parsing error: incorrect coverage value at line %1!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="495"/>
        <source>Skipped qualifier &apos;%1&apos; while saving a FPKM header.</source>
        <translation>Skipped qualifier &apos;%1&apos; while saving a FPKM header.</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="561"/>
        <source>FPKM Tracking Format saving error: tracking ID shouldn&apos;t be empty!</source>
        <translation>FPKM Tracking Format saving error: tracking ID shouldn&apos;t be empty!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="585"/>
        <source>FPKM Tracking Format saving error: failed to parse locus qualifier &apos;%1&apos;, writing it to the output file anyway!</source>
        <translation>FPKM Tracking Format saving error: failed to parse locus qualifier &apos;%1&apos;, writing it to the output file anyway!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="593"/>
        <source>FPKM Tracking Format saving error: an annotation region (%1, %2) differs from the information stored in the &apos;locus&apos; qualifier (%3, %4). Writing the &apos;locus&apos; qualifier to output!</source>
        <translation>FPKM Tracking Format saving error: an annotation region (%1, %2) differs from the information stored in the &apos;locus&apos; qualifier (%3, %4). Writing the &apos;locus&apos; qualifier to output!</translation>
    </message>
    <message>
        <location filename="../src/FpkmTrackingFormat.cpp" line="626"/>
        <source>FPKM Tracking Format saving error: one or more errors occurred while saving a file, see TRACE log for details!</source>
        <translation>FPKM Tracking Format saving error: one or more errors occurred while saving a file, see TRACE log for details!</translation>
    </message>
</context>
<context>
    <name>U2::GFFFormat</name>
    <message>
        <location filename="../src/GFFFormat.cpp" line="52"/>
        <source>GFF</source>
        <translation>GFF</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="53"/>
        <source>GFF is a format used for storing features and annotations</source>
        <translation>GFF это формат используемый для хранения аннотаций</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="94"/>
        <source>Parsing error: invalid header</source>
        <translation>Parsing error: invalid header</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="98"/>
        <source>Parsing error: file does not contain version header</source>
        <translation>Parsing error: file does not contain version header</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="102"/>
        <source>Parsing error: format version is not an integer</source>
        <translation>Parsing error: format version is not an integer</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="106"/>
        <source>Parsing error: GFF version %1 is not supported</source>
        <translation>Parsing error: GFF version %1 is not supported</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="218"/>
        <source>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</source>
        <translation>File &quot;%1&quot; contains too many sequences to be displayed. However, you can process these data using instruments from the menu &lt;i&gt;Tools -&gt; NGS data analysis&lt;/i&gt; or pipelines built with Workflow Designer.</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="273"/>
        <source>Parsing error: file contains empty line %1, line skipped</source>
        <translation>Parsing error: file contains empty line %1, line skipped</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="313"/>
        <source>Parsing error: sequence in FASTA sequence has whitespaces at line %1</source>
        <translation>Parsing error: sequence in FASTA sequence has whitespaces at line %1</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="320"/>
        <source>Parsing error: too few fields at line %1</source>
        <translation>Parsing error: too few fields at line %1</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="326"/>
        <source>Parsing error: start position at line %1 is not integer</source>
        <translation>Parsing error: start position at line %1 is not integer</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="332"/>
        <source>Parsing error: end position at line %1 is not integer</source>
        <translation>Parsing error: end position at line %1 is not integer</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="337"/>
        <source>Parsing error: incorrect annotation region at line %1</source>
        <translation>Parsing error: incorrect annotation region at line %1</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="388"/>
        <source>Parsing error: incorrect attributes field %1 at line %2</source>
        <translation>Parsing error: incorrect attributes field %1 at line %2</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="427"/>
        <source>Parsing error: incorrect score parameter at line %1. Score can be a float number or &apos;.&apos; symbol</source>
        <translation>Parsing error: incorrect score parameter at line %1. Score can be a float number or &apos;.&apos; symbol</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="437"/>
        <source>Parsing error: incorrect frame parameter at line %1. Frame can be a number between 0-2 or &apos;.&apos; symbol</source>
        <translation>Parsing error: incorrect frame parameter at line %1. Frame can be a number between 0-2 or &apos;.&apos; symbol</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="447"/>
        <source>Parsing error: incorrect strand patameter at line %1. Strand can be &apos;+&apos;,&apos;-&apos; or &apos;.&apos;</source>
        <translation>Parsing error: incorrect strand patameter at line %1. Strand can be &apos;+&apos;,&apos;-&apos; or &apos;.&apos;</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="505"/>
        <source>One or more sequences in this file don&apos;t have names. Their names are generated automatically.</source>
        <translation>One or more sequences in this file don&apos;t have names. Their names are generated automatically.</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="624"/>
        <source>Can not detect chromosome name. &apos;Chr&apos; name will be used.</source>
        <translation>Can not detect chromosome name. &apos;Chr&apos; name will be used.</translation>
    </message>
    <message>
        <location filename="../src/GFFFormat.cpp" line="378"/>
        <source>Wrong location for joined annotation at line %1. Line was skipped.</source>
        <translation>Wrong location for joined annotation at line %1. Line was skipped.</translation>
    </message>
</context>
<context>
    <name>U2::GTFFormat</name>
    <message>
        <location filename="../src/GTFFormat.cpp" line="100"/>
        <source>GTF</source>
        <translation>GTF</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="101"/>
        <source>The Gene transfer format (GTF) is a file format used to hold information about gene structure.</source>
        <translation>The Gene transfer format (GTF) это формат используемый для хранения информации о структуре гена.</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="155"/>
        <source>GTF parsing error: incorrect number of fields at line %1!</source>
        <translation>GTF parsing error: incorrect number of fields at line %1!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="160"/>
        <source>GTF parsing error: a field at line %1 is empty!</source>
        <translation>GTF parsing error: a field at line %1 is empty!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="165"/>
        <source>GTF parsing error: incorrect coordinates at line %1!</source>
        <translation>GTF parsing error: incorrect coordinates at line %1!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="189"/>
        <source>GTF parsing error: incorrect score value &quot;%1&quot; at line %2!</source>
        <translation>GTF parsing error: incorrect score value &quot;%1&quot; at line %2!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="198"/>
        <source>GTF parsing error: incorrect frame value &quot;%1&quot; at line %2!</source>
        <translation>GTF parsing error: incorrect frame value &quot;%1&quot; at line %2!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="219"/>
        <source>GTF parsing error: invalid attributes format at line %1!</source>
        <translation>GTF parsing error: invalid attributes format at line %1!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="227"/>
        <source>GTF parsing error: incorrect strand value &quot;%1&quot; at line %2!</source>
        <translation>GTF parsing error: incorrect strand value &quot;%1&quot; at line %2!</translation>
    </message>
    <message>
        <location filename="../src/GTFFormat.cpp" line="275"/>
        <source>File &quot;%1&quot; contains too many annotation tables to be displayed. However, you can process these data using pipelines built with Workflow Designer.</source>
        <translation>File &quot;%1&quot; contains too many annotation tables to be displayed. However, you can process these data using pipelines built with Workflow Designer.</translation>
    </message>
</context>
<context>
    <name>U2::Genbank::LocationParser</name>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="353"/>
        <location filename="../src/GenbankLocationParser.cpp" line="422"/>
        <source>&apos;a single base from a range&apos; in combination with &apos;sequence span&apos; is not supported</source>
        <translation>&apos;a single base from a range&apos; in combination with &apos;sequence span&apos; is not supported</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="356"/>
        <source>Ignoring &apos;&lt;&apos; at start position</source>
        <translation>Ignoring &apos;&lt;&apos; at start position</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="425"/>
        <source>Ignoring &apos;&gt;&apos; at end position</source>
        <translation>Ignoring &apos;&gt;&apos; at end position</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="488"/>
        <source>Wrong token after JOIN %1</source>
        <translation>Wrong token after JOIN %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="493"/>
        <source>Wrong token after JOIN  - order %1</source>
        <translation>Wrong token after JOIN  - order %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="502"/>
        <source>Can&apos;t parse location on JOIN</source>
        <translation>Can&apos;t parse location on JOIN</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="508"/>
        <location filename="../src/GenbankLocationParser.cpp" line="534"/>
        <location filename="../src/GenbankLocationParser.cpp" line="555"/>
        <location filename="../src/GenbankLocationParser.cpp" line="601"/>
        <source>Must be RIGHT_PARENTHESIS instead of %1</source>
        <translation>Must be RIGHT_PARENTHESIS instead of %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="514"/>
        <source>Wrong token after ORDER %1</source>
        <translation>Wrong token after ORDER %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="519"/>
        <source>Wrong token after ORDER - join %1</source>
        <translation>Wrong token after ORDER - join %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="528"/>
        <source>Can&apos;t parse location on ORDER</source>
        <translation>Can&apos;t parse location on ORDER</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="540"/>
        <source>Wrong token after BOND %1</source>
        <translation>Wrong token after BOND %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="549"/>
        <source>Can&apos;t parse location on BONDs</source>
        <translation>Can&apos;t parse location on BONDs</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="577"/>
        <source>Must be LEFT_PARENTHESIS instead of %1</source>
        <translation>Must be LEFT_PARENTHESIS instead of %1</translation>
    </message>
    <message>
        <location filename="../src/GenbankLocationParser.cpp" line="594"/>
        <source>Can&apos;t parse location on COMPLEMENT</source>
        <translation>Can&apos;t parse location on COMPLEMENT</translation>
    </message>
</context>
<context>
    <name>U2::GenbankPlainTextFormat</name>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="392"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="629"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="634"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="640"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="786"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="815"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="822"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="829"/>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="839"/>
        <source>Error writing document</source>
        <translation>Ошибка записи</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="48"/>
        <source>GenBank</source>
        <translation>GenBank</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="49"/>
        <source>GenBank Flat File Format is a rich format for storing sequences and associated annotations</source>
        <translation>GenBank Flat File Format это формат для хранения последовательностей и их аннотаций</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="92"/>
        <source>LOCUS is not the first line</source>
        <translation>Строка локуса должна идти первой</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="107"/>
        <source>Error parsing LOCUS line</source>
        <translation>Ошибка чтения локуса</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="172"/>
        <source>incomplete SOURCE record</source>
        <translation>Данные повреждены: запись SOURCE</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="526"/>
        <source>There is no IOAdapter registry yet</source>
        <translation>There is no IOAdapter registry yet</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="528"/>
        <source>IOAdapterFactory is NULL</source>
        <translation>IOAdapterFactory is NULL</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="530"/>
        <source>IOAdapter is NULL</source>
        <translation>IOAdapter is NULL</translation>
    </message>
    <message>
        <location filename="../src/GenbankPlainTextFormat.cpp" line="798"/>
        <source>Invalid annotation table!</source>
        <translation>Invalid annotation table!</translation>
    </message>
</context>
<context>
    <name>U2::GzipDecompressTask</name>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="125"/>
        <source>Decompression task</source>
        <translation>Задача разархивации</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="129"/>
        <source>&apos;%1&apos; is not zipped file</source>
        <translation>&apos;%1&apos; не является zip файлом</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="134"/>
        <source>Start decompression &apos;%1&apos;</source>
        <translation>Начало разархивации &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="136"/>
        <source>IOAdapterRegistry is NULL!</source>
        <translation>IOAdapterRegistry is NULL!</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="139"/>
        <location filename="../src/tasks/BgzipTask.cpp" line="141"/>
        <source>IOAdapterFactory is NULL!</source>
        <translation>IOAdapterFactory is NULL!</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="144"/>
        <location filename="../src/tasks/BgzipTask.cpp" line="147"/>
        <source>Can not create IOAdapter!</source>
        <translation>Can not create IOAdapter!</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="159"/>
        <source>Can not open output file &apos;%1&apos;</source>
        <translation>Невозможно открыть выходной файл &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="165"/>
        <source>Can not open input file &apos;%1&apos;</source>
        <translation>Can not open input file &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="180"/>
        <source>Error reading file</source>
        <translation>Ошибка чтения файла</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="186"/>
        <source>Error writing to file</source>
        <translation>Ошибка записи в файл</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="191"/>
        <source>Decompression finished</source>
        <translation>Разархивация завершена</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="196"/>
        <source>Decompression task was finished with an error: %1</source>
        <translation>Задача разархивации завершилась с ошибкой : %1</translation>
    </message>
    <message>
        <location filename="../src/tasks/BgzipTask.cpp" line="198"/>
        <source>Decompression task was finished. A new decompressed file is: &lt;a href=&quot;%1&quot;&gt;%1&lt;/a&gt;</source>
        <translation>Задача разархивации завершена. Новый файл: &lt;a href=&quot;%1&quot;&gt;%1&lt;/a&gt;</translation>
    </message>
</context>
<context>
    <name>U2::InfoPartParser</name>
    <message>
        <location filename="../src/util/SnpeffInfoParser.cpp" line="111"/>
        <source>Too few values in the entry: &apos;%1&apos;. Expected at least %2 values.</source>
        <translation>Too few values in the entry: &apos;%1&apos;. Expected at least %2 values.</translation>
    </message>
    <message>
        <location filename="../src/util/SnpeffInfoParser.cpp" line="131"/>
        <source>Too many values in the entry &apos;%1&apos;, extra entries are ignored</source>
        <translation>Too many values in the entry &apos;%1&apos;, extra entries are ignored</translation>
    </message>
</context>
<context>
    <name>U2::LoadConvertAndSaveSnpeffVariationsToAnnotationsTask</name>
    <message>
        <location filename="../src/tasks/ConvertSnpeffVariationsToAnnotationsTask.cpp" line="118"/>
        <source>Load file and convert SnpEff variations to annotations task</source>
        <translation>Load file and convert SnpEff variations to annotations task</translation>
    </message>
    <message>
        <location filename="../src/tasks/ConvertSnpeffVariationsToAnnotationsTask.cpp" line="157"/>
        <source>&apos;%1&apos; load failed, the result document is NULL</source>
        <translation>&apos;%1&apos; load failed, the result document is NULL</translation>
    </message>
    <message>
        <location filename="../src/tasks/ConvertSnpeffVariationsToAnnotationsTask.cpp" line="161"/>
        <source>File &apos;%1&apos; doesn&apos;t contain variation tracks</source>
        <translation>File &apos;%1&apos; doesn&apos;t contain variation tracks</translation>
    </message>
</context>
<context>
    <name>U2::MSFFormat</name>
    <message>
        <location filename="../src/MSFFormat.cpp" line="65"/>
        <source>MSF</source>
        <translation>MSF</translation>
    </message>
    <message>
        <location filename="../src/MSFFormat.cpp" line="67"/>
        <source>MSF format is used to store multiple aligned sequences. Files include the sequence name and the sequence itself, which is usually aligned with other sequences in the file.</source>
        <translation>MSF формат используется для множественных выравниваний. Файлы включают имя последовательности и последовательность, которая выровнена с другими последовательностями в файле.</translation>
    </message>
    <message>
        <location filename="../src/MSFFormat.cpp" line="138"/>
        <source>Incorrect format</source>
        <translation>Неверный формат</translation>
    </message>
    <message>
        <location filename="../src/MSFFormat.cpp" line="159"/>
        <source>Unexpected end of file</source>
        <translation>Неожиданный конец файла</translation>
    </message>
    <message>
        <location filename="../src/MSFFormat.cpp" line="195"/>
        <source>File check sum is incorrect: expected value: %1, current value %2</source>
        <translation>Контрольная сумма некорректна: ожидается значение %1, текущее значение %2</translation>
    </message>
    <message>
        <location filename="../src/MSFFormat.cpp" line="269"/>
        <source>Unexpected check sum in the row number %1, name: %2; expected value: %3, current value %4</source>
        <translation>Некорректная контрольная сумма в ряду %1, имя: %2; ожидается значение: %3, текущее значение %4</translation>
    </message>
    <message>
        <location filename="../src/MSFFormat.cpp" line="276"/>
        <source>Alphabet unknown</source>
        <translation>Неизвестный алфавит</translation>
    </message>
</context>
<context>
    <name>U2::MegaFormat</name>
    <message>
        <location filename="../src/MegaFormat.cpp" line="54"/>
        <source>Mega</source>
        <translation>Mega</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="55"/>
        <source>Mega is a file format of native MEGA program</source>
        <translation>Mega это формат файла программы MEGA</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="316"/>
        <source>Found sequences of different sizes</source>
        <translation>Обнаружены последовательности разной длины</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="153"/>
        <source>Bad name of sequence</source>
        <translation>Неверное имя последовательности</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="173"/>
        <source>Unexpected # in comments</source>
        <translation>Неожиданный символ &quot;#&quot; в комментариях</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="182"/>
        <source>A comment has not end</source>
        <translation>У комментария отсутствует окончание</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="206"/>
        <source>Unexpected symbol between comments</source>
        <translation>Неоижданные символы между комментариями</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="264"/>
        <source>Incorrect format</source>
        <translation>Неверный формат</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="277"/>
        <source>Identical symbol at the first sequence</source>
        <translation>Идентичный символ в первой последовательности</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="298"/>
        <source>Incorrect order of sequences&apos; names</source>
        <translation>Неправильный порядок имён последовательностей</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="304"/>
        <source>Incorrect sequence</source>
        <translation>Incorrect sequence</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="324"/>
        <source>Alphabet is unknown</source>
        <translation>Неизвестный алфавит</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="395"/>
        <location filename="../src/MegaFormat.cpp" line="402"/>
        <source>No header</source>
        <translation>Отсутствует заголовок</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="396"/>
        <source>No # before header</source>
        <translation>Отсутствует символ &quot;#&quot; перед заголовком</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="403"/>
        <source>Not MEGA-header</source>
        <translation>Не является заголовком MEGA</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="426"/>
        <location filename="../src/MegaFormat.cpp" line="435"/>
        <location filename="../src/MegaFormat.cpp" line="460"/>
        <source>No data in file</source>
        <translation>Отсутствуют данные в файле</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="441"/>
        <location filename="../src/MegaFormat.cpp" line="446"/>
        <source>Incorrect title</source>
        <translation>Неправильный заголовок</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="492"/>
        <source>Sequence has empty part</source>
        <translation>Часть последовательности пустая</translation>
    </message>
    <message>
        <location filename="../src/MegaFormat.cpp" line="512"/>
        <source>Bad symbols in a sequence</source>
        <translation>Некорректные символы в последовательности</translation>
    </message>
</context>
<context>
    <name>U2::MysqlUpgradeTask</name>
    <message>
        <location filename="../src/tasks/MysqlUpgradeTask.cpp" line="33"/>
        <source>Upgrade mysql database</source>
        <translation>Upgrade mysql database</translation>
    </message>
</context>
<context>
    <name>U2::NEXUSFormat</name>
    <message>
        <location filename="../src/NEXUSFormat.cpp" line="53"/>
        <source>NEXUS</source>
        <translation>NEXUS</translation>
    </message>
    <message>
        <location filename="../src/NEXUSFormat.cpp" line="54"/>
        <source>Nexus is a multiple alignment and phylogenetic trees file format</source>
        <translation>Nexus формат файла для множественных выравниваний и филогенетических деревьев</translation>
    </message>
    <message>
        <location filename="../src/NEXUSFormat.cpp" line="727"/>
        <source>#NEXUS header missing</source>
        <translation>Отсутствует заголовок #NEXUS</translation>
    </message>
</context>
<context>
    <name>U2::NewickFormat</name>
    <message>
        <location filename="../src/NewickFormat.cpp" line="44"/>
        <source>Newick Standard</source>
        <translation>Стандарт Newick</translation>
    </message>
    <message>
        <location filename="../src/NewickFormat.cpp" line="45"/>
        <source>Newick is a simple format used to write out trees in a text file</source>
        <translation>Newick iэто формат используемый для записи деревьев в текстовый файл</translation>
    </message>
</context>
<context>
    <name>U2::PDBFormat</name>
    <message>
        <location filename="../src/PDBFormat.cpp" line="202"/>
        <source>Line is too long</source>
        <translation>Слишком длинная строка</translation>
    </message>
    <message>
        <location filename="../src/PDBFormat.cpp" line="57"/>
        <source>The Protein Data Bank (PDB) format provides a standard representation for macromolecular structure data derived from X-ray diffraction and NMR studies.</source>
        <translation>The Protein Data Bank (PDB) формат обеспечивает стандартное представление для данных высокомолекулярных структур, полученных из рентгеновской дифракции и исследований ЯМР.</translation>
    </message>
    <message>
        <location filename="../src/PDBFormat.cpp" line="261"/>
        <source>Some mandatory records are absent</source>
        <translation>Некоторые обязательные записи отсутствуют</translation>
    </message>
    <message>
        <location filename="../src/PDBFormat.cpp" line="399"/>
        <source>PDB warning: unknown residue name: %1</source>
        <translation>PDB предупреждение: неизвестное имя остатка: %1</translation>
    </message>
    <message>
        <location filename="../src/PDBFormat.cpp" line="484"/>
        <source>Invalid secondary structure record</source>
        <translation>Неверная запись вторичной структуры</translation>
    </message>
    <message>
        <location filename="../src/PDBFormat.cpp" line="518"/>
        <source>Invalid SEQRES: less then 24 characters</source>
        <translation>Неверный SEQRES: меньше чем 24 символа</translation>
    </message>
    <message>
        <location filename="../src/PDBFormat.cpp" line="56"/>
        <source>PDB</source>
        <translation>PDB</translation>
    </message>
</context>
<context>
    <name>U2::PDWFormat</name>
    <message>
        <location filename="../src/PDWFormat.cpp" line="56"/>
        <source>pDRAW</source>
        <translation>pDRAW</translation>
    </message>
    <message>
        <location filename="../src/PDWFormat.cpp" line="57"/>
        <source>pDRAW is a sequence file format used by pDRAW software</source>
        <translation>pDRAW это формат файла для хранения последовательности используемый инструментом pDRAW</translation>
    </message>
    <message>
        <location filename="../src/PDWFormat.cpp" line="97"/>
        <location filename="../src/PDWFormat.cpp" line="187"/>
        <source>Line is too long</source>
        <translation>Слишком длинная строка</translation>
    </message>
</context>
<context>
    <name>U2::PairedFastqComparator</name>
    <message>
        <location filename="../src/util/PairedFastqComparator.cpp" line="71"/>
        <source>Too much reads without a pair (&gt;%1). Check the input data are set correctly.</source>
        <translation>Too much reads without a pair (&gt;%1). Check the input data are set correctly.</translation>
    </message>
    <message>
        <location filename="../src/util/PairedFastqComparator.cpp" line="165"/>
        <source>Invalid sequence info</source>
        <translation>Invalid sequence info</translation>
    </message>
</context>
<context>
    <name>U2::PhylipFormat</name>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="48"/>
        <source>PHYLIP multiple alignment format for phylogenetic applications.</source>
        <translation>PHYLIP формат филогенетических деревьев.</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="77"/>
        <source>Alphabet is unknown</source>
        <translation>Неизвестный алфавит</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="351"/>
        <source>Error parsing file</source>
        <translation>Ошибка разбора формата</translation>
    </message>
</context>
<context>
    <name>U2::PhylipInterleavedFormat</name>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="239"/>
        <source>PHYLIP Interleaved</source>
        <translation>PHYLIP Interleaved</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="337"/>
        <source>Illegal line</source>
        <translation>Неправильная строка</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="344"/>
        <source>Wrong header</source>
        <translation>Неверный заголовок</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="384"/>
        <source>Block is incomplete</source>
        <translation>Block is incomplete</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="395"/>
        <source>Block is incomlete</source>
        <translation>Block is incomlete</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="403"/>
        <source>Number of characters does not correspond to the stated number</source>
        <translation>Число символов не соответствует установленному числу</translation>
    </message>
</context>
<context>
    <name>U2::PhylipSequentialFormat</name>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="119"/>
        <source>PHYLIP Sequential</source>
        <translation>PHYLIP Sequential</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="201"/>
        <source>Illegal line</source>
        <translation>Неправильная строка</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="208"/>
        <source>Wrong header</source>
        <translation>Неверный заголовок</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="211"/>
        <location filename="../src/PhylipFormat.cpp" line="348"/>
        <source>There is not enough data</source>
        <translation>Недостаточно данных</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="200"/>
        <location filename="../src/PhylipFormat.cpp" line="215"/>
        <location filename="../src/PhylipFormat.cpp" line="359"/>
        <source>Error parsing file</source>
        <translation>Ошибка разбора формата</translation>
    </message>
    <message>
        <location filename="../src/PhylipFormat.cpp" line="232"/>
        <source>Number of characters does not correspond to the stated number</source>
        <translation>Число символов не соответствует установленному числу</translation>
    </message>
</context>
<context>
    <name>U2::PlainTextFormat</name>
    <message>
        <location filename="../src/PlainTextFormat.cpp" line="38"/>
        <source>Plain text</source>
        <translation>Текст</translation>
    </message>
    <message>
        <location filename="../src/PlainTextFormat.cpp" line="40"/>
        <source>A simple plain text file.</source>
        <translation>Текстовый файл.</translation>
    </message>
</context>
<context>
    <name>U2::RawDNASequenceFormat</name>
    <message>
        <location filename="../src/RawDNASequenceFormat.cpp" line="47"/>
        <source>Raw sequence</source>
        <translation>Произвольная последовательность</translation>
    </message>
    <message>
        <location filename="../src/RawDNASequenceFormat.cpp" line="51"/>
        <source>Raw sequence file - a whole content of the file is treated either as a single/multiple nucleotide or peptide sequence(s). UGENE will remove all non-alphabetic chars from the result sequence. By default the characters in the file are considered a single sequence.</source>
        <translation>Сырой файл последовательности - все содержимое файла рассматривается как один нуклеотид или пептидная последовательность. UGENE удалит все символы не принадлежащие алфавиту из результирующей последовательности.</translation>
    </message>
    <message>
        <location filename="../src/RawDNASequenceFormat.cpp" line="130"/>
        <source>Sequence is empty</source>
        <translation>Последовательность пуста</translation>
    </message>
</context>
<context>
    <name>U2::SAMFormat</name>
    <message>
        <location filename="../src/SAMFormat.cpp" line="82"/>
        <source>Field &quot;%1&quot; not matched pattern &quot;%2&quot;, expected pattern &quot;%3&quot;</source>
        <translation>Field &quot;%1&quot; not matched pattern &quot;%2&quot;, expected pattern &quot;%3&quot;</translation>
    </message>
    <message>
        <location filename="../src/SAMFormat.cpp" line="91"/>
        <source>SAM</source>
        <translation>SAM</translation>
    </message>
    <message>
        <location filename="../src/SAMFormat.cpp" line="92"/>
        <source>The Sequence Alignment/Map (SAM) format is a generic alignment format forstoring read alignments against reference sequence</source>
        <translation>The Sequence Alignment/Map (SAM) это общий формат выравниваний для хранения выравниваний с референсной последовательностью</translation>
    </message>
</context>
<context>
    <name>U2::SCFFormat</name>
    <message>
        <location filename="../src/SCFFormat.cpp" line="51"/>
        <source>SCF</source>
        <translation>SCF</translation>
    </message>
    <message>
        <location filename="../src/SCFFormat.cpp" line="52"/>
        <source>It is Standard Chromatogram Format</source>
        <translation>Это стандартный формат хроматограмм</translation>
    </message>
    <message>
        <location filename="../src/SCFFormat.cpp" line="70"/>
        <source>Failed to parse SCF file: %1</source>
        <translation>Failed to parse SCF file: %1</translation>
    </message>
    <message>
        <location filename="../src/SCFFormat.cpp" line="1239"/>
        <source>Failed to load sequence from SCF file %1</source>
        <translation>Невозможно загрузить последовательность из SCF файла %1</translation>
    </message>
</context>
<context>
    <name>U2::SnpeffInfoParser</name>
    <message>
        <location filename="../src/util/SnpeffInfoParser.cpp" line="47"/>
        <source>Can&apos;t parse the next INFO part: &apos;%1&apos;</source>
        <translation>Can&apos;t parse the next INFO part: &apos;%1&apos;</translation>
    </message>
</context>
<context>
    <name>U2::StockholmFormat</name>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="728"/>
        <source>Stockholm</source>
        <translation>Stockholm</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="729"/>
        <source>A multiple sequence alignments file format</source>
        <translation>Формат файла для множественных выравниваний</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="747"/>
        <location filename="../src/StockholmFormat.cpp" line="764"/>
        <source>unknown error occurred</source>
        <translation>Ошибка</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="493"/>
        <source>invalid file: bad header line</source>
        <translation>Неверный заголовок</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="531"/>
        <source>invalid file: empty sequence name</source>
        <translation>Не указано имя последовательности</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="534"/>
        <source>invalid file: equal sequence names in one block</source>
        <translation>одинаковые имена последовательностей в блоке</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="545"/>
        <source>invalid file: sequence names are not equal in blocks</source>
        <translation>Непарное имя последовательности в блоке</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="541"/>
        <location filename="../src/StockholmFormat.cpp" line="551"/>
        <source>invalid file: sequences in block are not of equal size</source>
        <translation>Блок содержит последовательности разной длины</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="570"/>
        <source>invalid file: empty sequence alignment</source>
        <translation>Выравнивание не содержит последовательностей</translation>
    </message>
    <message>
        <location filename="../src/StockholmFormat.cpp" line="574"/>
        <source>invalid file: unknown alphabet</source>
        <translation>Не удалось установить алфавит</translation>
    </message>
</context>
<context>
    <name>U2::StreamSequenceReader</name>
    <message>
        <location filename="../src/StreamSequenceReader.cpp" line="92"/>
        <source>File %1 unsupported format.</source>
        <translation>File %1 unsupported format.</translation>
    </message>
    <message>
        <location filename="../src/StreamSequenceReader.cpp" line="110"/>
        <source>Unsupported file format or short reads list is empty</source>
        <translation>Unsupported file format or short reads list is empty</translation>
    </message>
</context>
<context>
    <name>U2::SwissProtPlainTextFormat</name>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="63"/>
        <source>Swiss-Prot</source>
        <translation>Swiss-Prot</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="64"/>
        <source>SwissProt is a format of the UniProtKB/Swiss-prot database used for storing annotated protein sequence</source>
        <translation>SwissProt это формат базы данных UniProtKB/Swiss-prot используемый для хранения аннотированных белковых последовательностей</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="105"/>
        <source>ID is not the first line</source>
        <translation>Строка идентификатора должна идти первой</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="112"/>
        <source>Error parsing ID line</source>
        <translation>Неверный заголовок</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="122"/>
        <source>Error parsing ID line. Not found sequence length</source>
        <translation>Ошибка распознавания ID строки. Не найдена длина последовательности</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="239"/>
        <source>Reading sequence %1</source>
        <translation>Чтение последовательности: %1</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="258"/>
        <source>Error parsing sequence: unexpected empty line</source>
        <translation>Ошибка чтения последовательности: пустая строка</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="282"/>
        <source>Error reading sequence: memory allocation failed</source>
        <translation>Ошибка чтения последовательности: не удалось выделить память</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="297"/>
        <source>Sequence is truncated</source>
        <translation>Последовательность повреждена</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="304"/>
        <source>Reading annotations %1</source>
        <translation>Чтение аннотаций: %1</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="311"/>
        <source>Invalid format of feature table</source>
        <translation>Таблица аннотаций повреждена</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="343"/>
        <source>The DT string doesn&apos;t contain date.</source>
        <translation>The DT string doesn&apos;t contain date.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="347"/>
        <source>The format of the date is unexpected.</source>
        <translation>The format of the date is unexpected.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="351"/>
        <source>Day is incorrect.</source>
        <translation>Day is incorrect.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="354"/>
        <source>Mounth is incorrect.</source>
        <translation>Mounth is incorrect.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="357"/>
        <source>Year is incorrect.</source>
        <translation>Year is incorrect.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="398"/>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="486"/>
        <source>The annotation start position is unexpected.</source>
        <translation>The annotation start position is unexpected.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="401"/>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="491"/>
        <source>The annotation end position is unexpected.</source>
        <translation>The annotation end position is unexpected.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="474"/>
        <source>Unexpected annotation strings.</source>
        <translation>Unexpected annotation strings.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="481"/>
        <source>The annotation name is empty.</source>
        <translation>The annotation name is empty.</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="504"/>
        <source>Annotation qualifier is corrupted</source>
        <translation>Annotation qualifier is corrupted</translation>
    </message>
    <message>
        <location filename="../src/SwissProtPlainTextFormat.cpp" line="516"/>
        <source>Unexpected qulifiers values.</source>
        <translation>Unexpected qulifiers values.</translation>
    </message>
</context>
<context>
    <name>U2::TabulatedFormatReader</name>
    <message>
        <location filename="../src/util/TabulatedFormatReader.cpp" line="35"/>
        <source>IO adapter is not opened</source>
        <translation>IO adapter is not opened</translation>
    </message>
</context>
<context>
    <name>U2::U2DbiL10n</name>
    <message>
        <location filename="../src/mysql_dbi/MysqlAssemblyDbi.cpp" line="89"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="81"/>
        <source>There is no assembly object with the specified id.</source>
        <translation>Не найден объект сборки с указанным идентификатором.</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlAttributeDbi.cpp" line="224"/>
        <location filename="../src/sqlite_dbi/SQLiteAttributeDbi.cpp" line="237"/>
        <source>Unsupported attribute type: %1</source>
        <translation>Неподдерживаемый тип атрибута: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlAttributeDbi.cpp" line="357"/>
        <source>Required attribute is not found</source>
        <translation>Требуемый атрибут не найдет</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="272"/>
        <source>Database url is incorrect</source>
        <translation>Неправильный путь базы данных</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="278"/>
        <source>User login is not specified</source>
        <translation>Не указано имя пользователя</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="284"/>
        <source>Host is not specified</source>
        <translation>Не указан хост</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="290"/>
        <source>Database name is not specified</source>
        <translation>Не указано имя базы данных</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="306"/>
        <source>Error opening MySQL database: %1</source>
        <translation>Ошибка открытия базы данных MySQL: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="387"/>
        <source>Not a %1 MySQL database: %2, %3</source>
        <translation>Не %1 MySQL база данных: %2, %3</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="395"/>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="273"/>
        <source>Warning! The database was created with a newer %1 version: %2. Not all database features may be supported! Current %1 version: %3.</source>
        <translation>Предупреждение! База данных была создана с более новой %1 версией: %2. Не все возможности базы данных могут быть выполнены! Текущая %1 версия: %3.</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="500"/>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="330"/>
        <source>Illegal database state: %1</source>
        <translation>Неправильное состояние базы данных: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlDbi.cpp" line="530"/>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="412"/>
        <source>Can&apos;t synchronize database state</source>
        <translation>Невозможно синхронизовать состояние базы данных</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlFeatureDbi.cpp" line="155"/>
        <source>Annotation table object is not found.</source>
        <translation>Не найдена таблица аннотаций.</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlFeatureDbi.cpp" line="186"/>
        <source>Feature is not found.</source>
        <translation>Аннотация не найдена.</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="145"/>
        <source>An object single modification step not found</source>
        <translation>Объект единичной модификации не найден</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="181"/>
        <source>Failed to find user step ID</source>
        <translation>Невозможно найти идентификатор объекта единичной модификации</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="389"/>
        <source>Not main thread</source>
        <translation>Не основной поток</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="401"/>
        <source>Can&apos;t create a common user modifications step, previous one is not complete</source>
        <translation>Невозможно создать новую единичную модификацию, т.к. прошлая модификация еще не завершена</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="464"/>
        <source>Can&apos;t create a common multiple modifications step, previous one is not complete</source>
        <translation>Невозможно создать новые модификации, т.к. прошлая модификация еще не завершена</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="500"/>
        <source>Failed to create a common user modifications step</source>
        <translation>Не удалось создать новую единичную модификацию</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlModDbi.cpp" line="521"/>
        <source>Failed to create a common multiple modifications step</source>
        <translation>Не удалось создать новые модификации</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="111"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="127"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="226"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="645"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="390"/>
        <source>Msa object not found</source>
        <translation>Объект msa не найден</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="187"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="821"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="861"/>
        <source>Msa row not found</source>
        <translation>Столбец msa не найден</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="606"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="631"/>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1142"/>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1151"/>
        <source>Unexpected modification type &apos;%1&apos;</source>
        <translation>Неожиданный тип модификации &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="696"/>
        <source>Invalid row position: %1</source>
        <translation>Неправильное расположение столбца: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1070"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1201"/>
        <source>An error occurred during updating an alignment alphabet</source>
        <translation>Возникла ошибка во время обновления алфавита выравнивания</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1088"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1216"/>
        <source>An error occurred during reverting adding of rows</source>
        <translation>Возникла ошибка во время отмены добавления столбцов</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1106"/>
        <source>An error occurred during reverting addition of a row</source>
        <translation>Возникла ошибка во время отмены добавления столбца</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1119"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1236"/>
        <source>An error occurred during reverting removing of rows</source>
        <translation>Возникла ошибка во время отмены удаления столбцов</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1132"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1251"/>
        <source>An error occurred during reverting removing of a row</source>
        <translation>Возникла ошибка во время отмены удаления столбца</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1146"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1262"/>
        <source>An error occurred during updating an alignment gaps</source>
        <translation>Возникла ошибка во время обновления пробелов выравнивания</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1159"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1272"/>
        <source>An error occurred during updating an alignment row order</source>
        <translation>Возникла ошибка во время обновления порядка столбцов</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1173"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1282"/>
        <source>An error occurred during updating a row info</source>
        <translation>Возникла ошибка во время обновления информации о столбце</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1188"/>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1297"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="1323"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="1333"/>
        <source>An error occurred during updating an msa length</source>
        <translation>An error occurred during updating an msa length</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlMsaDbi.cpp" line="1226"/>
        <source>An error occurred during addition of a row</source>
        <translation>Возникла ошибка во время добавления столбца</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="608"/>
        <source>Can&apos;t undo an operation for the object</source>
        <translation>Невозможно отменить операцию для объекта</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="671"/>
        <source>Can&apos;t redo an operation for the object</source>
        <translation>Невозможно передвинуться на одну операцию вперед для объекта</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="812"/>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="918"/>
        <source>Object not found</source>
        <translation>Объект не найден</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="866"/>
        <source>Folder not found: %1 (canonical: %2)</source>
        <translation>Папка не найдена: %1 (каноническое: %2)</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1050"/>
        <source>Not an object, id: %1, type: %2</source>
        <translation>Не является объектом, id: %1, тип: %2</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1074"/>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="302"/>
        <source>Unknown object type! Id: %1, type: %2</source>
        <translation>Unknown object type! Id: %1, type: %2</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1118"/>
        <source>Can&apos;t undo an unknown operation: &apos;%1&apos;</source>
        <translation>Can&apos;t undo an unknown operation: &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1133"/>
        <source>Can&apos;t redo an unknown operation: &apos;%1&apos;</source>
        <translation>Can&apos;t redo an unknown operation: &apos;%1&apos;</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1164"/>
        <source>An error occurred during updating an object name</source>
        <translation>An error occurred during updating an object name</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="1182"/>
        <source>An error occurred during updating an object name!</source>
        <translation>An error occurred during updating an object name!</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlSequenceDbi.cpp" line="79"/>
        <source>Sequence object not found</source>
        <translation>Последовательность не найдена</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlSequenceDbi.cpp" line="131"/>
        <source>Internal error occurred during the sequence processing</source>
        <translation>Internal error occurred during the sequence processing</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlSequenceDbi.cpp" line="132"/>
        <source>An exception was thrown during reading sequence data from dbi</source>
        <translation>An exception was thrown during reading sequence data from dbi</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlSequenceDbi.cpp" line="395"/>
        <source>An error occurred during reverting replacing sequence data</source>
        <translation>An error occurred during reverting replacing sequence data</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlSequenceDbi.cpp" line="410"/>
        <source>An error occurred during replacing sequence data</source>
        <translation>An error occurred during replacing sequence data</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlVariantDbi.cpp" line="164"/>
        <source>Invalid variant track type: %1</source>
        <translation>Invalid variant track type: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlVariantDbi.cpp" line="193"/>
        <location filename="../src/mysql_dbi/MysqlVariantDbi.cpp" line="226"/>
        <source>Sequence name is not set</source>
        <translation>Sequence name is not set</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlVariantDbi.cpp" line="328"/>
        <source>New variant public ID is empty</source>
        <translation>New variant public ID is empty</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlVariantDbi.cpp" line="344"/>
        <source>New variant track ID is empty</source>
        <translation>New variant track ID is empty</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="124"/>
        <source>Packed data are empty</source>
        <translation>Packed data are empty</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="131"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="459"/>
        <source>Packing method prefix is not supported: %1</source>
        <translation>Packing method prefix is not supported: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="139"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="467"/>
        <source>Data are corrupted, no name end marker found: %1</source>
        <translation>Data are corrupted, no name end marker found: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="148"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="476"/>
        <source>Data are corrupted, no sequence end marker found: %1</source>
        <translation>Data are corrupted, no sequence end marker found: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="157"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="485"/>
        <source>Data are corrupted, no CIGAR end marker found: %1</source>
        <translation>Data are corrupted, no CIGAR end marker found: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="176"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="504"/>
        <source>Data are corrupted, no rnext end marker found: %1</source>
        <translation>Data are corrupted, no rnext end marker found: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlAssemblyUtils.cpp" line="191"/>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="519"/>
        <source>Can not convert pnext to a number: %1</source>
        <translation>Can not convert pnext to a number: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlHelpers.cpp" line="406"/>
        <source>Bound values: </source>
        <translation>Связанные значения: </translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlHelpers.cpp" line="426"/>
        <source>Cannot start a transaction</source>
        <translation>Cannot start a transaction</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlMultiTableAssemblyAdapter.cpp" line="502"/>
        <location filename="../src/sqlite_dbi/assembly/MultiTableAssemblyAdapter.cpp" line="125"/>
        <source>Failed to detect assembly storage format: %1</source>
        <translation>Failed to detect assembly storage format: %1</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlMultiTableAssemblyAdapter.cpp" line="514"/>
        <source>Failed to parse range: %1, full: %2</source>
        <translation>Failed to parse range: %1, full: %2</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/util/MysqlMultiTableAssemblyAdapter.cpp" line="528"/>
        <location filename="../src/mysql_dbi/util/MysqlMultiTableAssemblyAdapter.cpp" line="533"/>
        <location filename="../src/mysql_dbi/util/MysqlMultiTableAssemblyAdapter.cpp" line="539"/>
        <location filename="../src/sqlite_dbi/assembly/MultiTableAssemblyAdapter.cpp" line="149"/>
        <location filename="../src/sqlite_dbi/assembly/MultiTableAssemblyAdapter.cpp" line="153"/>
        <location filename="../src/sqlite_dbi/assembly/MultiTableAssemblyAdapter.cpp" line="158"/>
        <source>Failed to parse packed row range info %1</source>
        <translation>Failed to parse packed row range info %1</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="96"/>
        <source>Unsupported reads storage type: %1</source>
        <translation>Unsupported reads storage type: %1</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="364"/>
        <source>Packing method is not supported: %1</source>
        <translation>Packing method is not supported: %1</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteAssemblyDbi.cpp" line="452"/>
        <source>Packed data are empty!</source>
        <translation>Packed data are empty!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="223"/>
        <source>Error checking SQLite database: %1!</source>
        <translation>Error checking SQLite database: %1!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="236"/>
        <source>Error creating table: %1, error: %2</source>
        <translation>Error creating table: %1, error: %2</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="268"/>
        <source>Not a %1 SQLite database: %2</source>
        <translation>Not a %1 SQLite database: %2</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="326"/>
        <source>Database is already opened!</source>
        <translation>База данных уже открыта!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="336"/>
        <source>URL is not specified</source>
        <translation>Не задан путь</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="350"/>
        <source>Error opening SQLite database: %1!</source>
        <translation>Error opening SQLite database: %1!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="402"/>
        <source>Database is already closed!</source>
        <translation>База данных уже закрыта!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="406"/>
        <source>Illegal database state %1!</source>
        <translation>Illegal database state %1!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteDbi.cpp" line="431"/>
        <source>Failed to close database: %1, err: %2</source>
        <translation>Failed to close database: %1, err: %2</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteFeatureDbi.cpp" line="153"/>
        <source>Annotation table object not found.</source>
        <translation>Annotation table object not found.</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteModDbi.cpp" line="147"/>
        <source>An object single modification step not found!</source>
        <translation>An object single modification step not found!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="542"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="557"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="749"/>
        <source>Msa object not found!</source>
        <translation>Msa object not found!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="640"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="808"/>
        <location filename="../src/sqlite_dbi/SQLiteMsaDbi.cpp" line="845"/>
        <source>Msa row not found!</source>
        <translation>Msa row not found!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="278"/>
        <source>Not an object! Id: %1, type: %2</source>
        <translation>Not an object! Id: %1, type: %2</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="616"/>
        <source>Can&apos;t undo an operation for the object!</source>
        <translation>Can&apos;t undo an operation for the object!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="699"/>
        <source>Can&apos;t redo an operation for the object!</source>
        <translation>Can&apos;t redo an operation for the object!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="870"/>
        <source>Object not found!</source>
        <translation>Object not found!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="1004"/>
        <source>Unexpected row count. Query: &apos;%1&apos;, rows: %2</source>
        <translation>Unexpected row count. Query: &apos;%1&apos;, rows: %2</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlObjectDbi.cpp" line="824"/>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="930"/>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="942"/>
        <source>Object not found.</source>
        <translation>Object not found.</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteObjectDbi.cpp" line="979"/>
        <source>Folder not found: %1</source>
        <translation>Folder not found: %1</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteSequenceDbi.cpp" line="73"/>
        <source>Sequence object not found.</source>
        <translation>Sequence object not found.</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/SQLiteVariantDbi.cpp" line="103"/>
        <location filename="../src/sqlite_dbi/SQLiteVariantDbi.cpp" line="136"/>
        <source>Sequence name is not set!</source>
        <translation>Sequence name is not set!</translation>
    </message>
    <message>
        <location filename="../src/sqlite_dbi/assembly/RTreeAssemblyAdapter.cpp" line="64"/>
        <source>Error during RTree index creation: %1! Check if SQLite library has RTree index support!</source>
        <translation>Error during RTree index creation: %1! Check if SQLite library has RTree index support!</translation>
    </message>
    <message>
        <location filename="../src/mysql_dbi/MysqlUdrDbi.cpp" line="61"/>
        <location filename="../src/mysql_dbi/MysqlUdrDbi.cpp" line="75"/>
        <location filename="../src/sqlite_dbi/SQLiteUdrDbi.cpp" line="54"/>
        <location filename="../src/sqlite_dbi/SQLiteUdrDbi.cpp" line="65"/>
        <source>An error occurred during updating UDR</source>
        <translation>An error occurred during updating UDR</translation>
    </message>
</context>
<context>
    <name>U2::VectorNtiSequenceFormat</name>
    <message>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="46"/>
        <source>Vector NTI sequence</source>
        <translation>Последовательность Vector NTI</translation>
    </message>
    <message>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="47"/>
        <source>Vector NTI sequence format is a rich format based on NCBI GenBank format for storing sequences and associated annotations</source>
        <translation>Формат Vector NTI это формат основанный на формате NCBI GenBank для хранения последовательностей и аннотаций</translation>
    </message>
    <message>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="298"/>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="314"/>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="318"/>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="323"/>
        <location filename="../src/VectorNtiSequenceFormat.cpp" line="329"/>
        <source>Error writing document</source>
        <translation>Ошибка записи</translation>
    </message>
</context>
</TS>
