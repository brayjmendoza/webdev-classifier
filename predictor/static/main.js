function correction(classificationType) {
    // Send correction if prediction was wrong
    // classificationType is the thing that was predicted (i.e. iris species)

    let path = `/${classificationType}/incorrect`;

    document.getElementById('correction').addEventListener('submit', function(e) {
        e.preventDefault();
        const correction = document.getElementById('species').value;
            
        fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ correction })
        })
        .then(response => {
                if (response.ok) {
                    document.getElementById('correction').style.display='none';
                    document.getElementById('feedback').innerText = 'Thanks for the feedback!';
                } else {
                    // Handle non 200 responses
                    document.getElementById('correction').style.display='none';
                    document.getElementById('feedback').innerText = 'An error occurred';
                    document.getElementById('tryAgain').style.display='block';
                } 
        })
        .catch(error => {
                console.error('Error: ', error);
                document.getElementById('correction').style.display='none';
                document.getElementById('feedback').innerText = 'An error occurred';
                document.getElementById('tryAgain').style.display='block';
        })
    });
}