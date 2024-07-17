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
            document.getElementById('correctionsPage').innerHTML = data;
            document.getElementById('retrainButton').style.display = 'block';  
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
        .then(response => response.text())
        .then(data => {
            document.getElementById('retrainButton').style.display = 'none';
            document.getElementById('retrainStatus').innerText = "Success!";
            document.getElementById('retrainPlots').innerHTML = data;
            console.log(data);

            // Prevent images from caching so they are properly updated in real time
            const images = document.getElementById('retrainPlots').getElementsByTagName('img');
            for (let i = 0; i < images.length; i++) {
                const img = images[i];
                const src = img.src.split('?')[0]; // Remove existing query parameters

                // Use timestamp as query parameter for unique URL
                img.src = src + '?t=' + new Date().getTime(); 
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('retrainStatus').innerText = "An error occurred.";
        });
    });
}