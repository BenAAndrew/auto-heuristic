$(".custom-file-input").on("change", function() {
    var fileName = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
});

const fileUpload = document.getElementById('file');
const columnSelector = document.getElementById('target');
fileUpload.addEventListener('change', (event) => {
    const [file] = event.target.files;
    const reader = new FileReader();

    reader.addEventListener("load", () => {
        const header = reader.result.split("\n")[0].trim();
        console.log(header);
        if (header.length == 0 || !header.includes(",")){
            alert("No header in CSV");
        }

        $('#target').empty();

        const columns = header.split(",");
        if (columns.length < 2){
            alert("Not enough columns in CSV");
        }
        for(const column of columns){
            columnSelector.appendChild(new Option(column, column));
        }
    }, false);

    if (file) {
        reader.readAsText(file);
    }
});

function download(filename, text) {
    var element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);

    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}
