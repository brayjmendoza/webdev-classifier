function correction(classType) {
    // Send correction if prediction was wrong
    // classType is the thing that was predicted (i.e. iris species)

    let path = `/${classType}/incorrect`;

    document.getElementById('correction').addEventListener('submit', function(e) {
        e.preventDefault(); // prevents reload
        const correction = document.getElementById('species').value;
            
        fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ correction })
        })
        .then(response => response.text())
        .then(data => {
            document.getElementById('correction').style.display='none';
            document.getElementById('feedback').innerText = 'Thanks for the feedback!';
            document.getElementById('cTable').innerHTML = data;
            document.addEventListener("DOMContentLoaded", function() {
                document.getElementById('retrainButton').style.display = 'block';
            });
            
        })
        .catch(error => {
            console.error('Error: ', error);
            document.getElementById('correction').style.display='none';
            document.getElementById('feedback').innerText = 'An error occurred';
            document.getElementById('tryAgain').style.display='block';
        })
    });
}

function retrain(classType, model) {
    /* Retrains a given model with data from database
     *
     * Parameters:
     * classType - the thing that is being predicted (e.g. iris species)
     * model - the model to be retrained (e.g. knn, dtree)
     */ 

    let path = `/${classType}/${model}/retrain_and_visualize`

    console.log(path)

    document.getElementById('retrainButton').addEventListener('click', function() {
        fetch(path, {
            method: 'POST'
        })
        .then(response => {
            document.getElementById('retrainButton').style.display = 'none';
            if(response.ok) {
                document.getElementById('retrainStatus').innerText = "Success!";
            } else {
                document.getElementById('retrainStatus').innerText = "An error occurred.";
            }
            
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('retrainStatus').innerText = "An error occurred.";
        });
    });
}