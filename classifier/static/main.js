function correction(classType) {
    /* Send correction if prediction was wrong
     * classType is the thing that was predicted (i.e. iris species)
     */

    let path = `/${classType}/incorrect`;

    document.getElementById('correction').addEventListener('submit', function(e) {
        e.preventDefault(); // prevents reload
        const correction = document.getElementById('species').value;
        
        // Store state of corrections table (for display purposes)
        let isExpanded;
        if (document.getElementById('correctionsTableContainer')) {
            isExpanded = document.getElementById('correctionsTableContainer').classList.contains('expanded');
        } else {
            isExpanded = false;
        }
            
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

            // Restore corrections list state
            const tableContainer = document.getElementById('correctionsTableContainer');
            tableContainer.style.transition = 'none'; // disable transition
            if (isExpanded) {
                const table = document.getElementById('correctionsTable');
                const caret = document.getElementById('caret');
                const clearButtonContainer = document.getElementById('clearCorrectionsContainer');

                tableContainer.classList.toggle('expanded');
                caret.classList.toggle('caret-up');
                tableContainer.style.height = table.scrollHeight + 'px';
                clearButtonContainer.style.display = 'inline-block';
            }
            void tableContainer.offsetHeight; // force reflow to apply changes without transition
            tableContainer.style.transition = ''; // re-enable transition
        })
        .catch(error => {
            console.error('Error: ', error);
            document.getElementById('correction').style.display='none';
            document.getElementById('feedback').innerText = 'An error occurred';
            document.getElementById('tryAgain').style.display='block';
            document.getElementById('retrainButton').style.display = 'none';
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
    
    document.addEventListener('DOMContentLoaded', () => {
        // Connect to the Socket.IO server
        const socket = io();

        // Handle the 'status' event from the server
        socket.on('retraining-status', function(data) {
            document.getElementById('retrainStatus').innerText = data.message;
        });
    
        let path = `/${classType}/${model}/retrain_and_visualize`
        document.getElementById('retrainButton').addEventListener('click', function() {
            document.getElementById('retrainButton').style.display = 'none';
            fetch(path, {
                method: 'POST'
            })
            .then(response => response.text())
            .then(data => {
                document.getElementById('retrainStatus').innerText = "Success!";
                document.getElementById('retrainPlots').innerHTML = data;

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
    });
}

// Handle corrections table dropdown functionality
function attachCorrectionListeners(classType, model) {
    function attachDropdownListener() {
        const tableHeader = document.getElementById('correctionsHeader');
        const tableContainer = document.getElementById('correctionsTableContainer');
        const table = document.getElementById('correctionsTable');
        const caret = document.getElementById('caret');
        const clearButtonContainer = document.getElementById('clearCorrectionsContainer');

        if (tableHeader && tableContainer && caret) {
            tableHeader.addEventListener('click', () => {
                tableContainer.classList.toggle('expanded');
                caret.classList.toggle('caret-up');

                let isExpanded = tableContainer.classList.contains('expanded');

                if (isExpanded) {
                    tableContainer.style.height = table.scrollHeight + 'px';
                    clearButtonContainer.style.display = 'inline-block';
                } else {
                    tableContainer.style.height = '0px';
                    clearButtonContainer.style.display = 'none';
                }
            });
        }
    }

    // Initial attachment of event listeners
    const tableHeader = document.getElementById('correctionsHeader');
    if (tableHeader)  {
        attachDropdownListener();
        clearCorrections(classType, model);
    }

    // Create a MutationObserver to watch for changes in the corrections container
    const correctionsListObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList' && mutation.addedNodes.length) {
                // Save the current visibility state of the table
                // const tableContainer = document.getElementById('correctionsTableContainer');
                // let isExpanded = tableContainer.classList.contains('expanded');

                // Reattach the dropdown listener for new content
                attachDropdownListener();
                clearCorrections(classType, model);

                // // Restore the visibility state of the table
                // if (isExpanded) {
                //     tableContainer.classList.toggle('expanded');
                // }
            }
        });
    });

    // Observe the corrections page for changes
    const correctionsContainer = document.getElementById('correctionsPage');
    correctionsListObserver.observe(correctionsContainer, {
        childList: true,  // Observe addition/removal of child nodes
        subtree: true     // Observe all descendant nodes
    });
}

function clearCorrections(classType, model) {
    /*
     * Clear corrections table and corresponding data
     */

    // Connect to the Socket.IO server
    const socket = io();

    // Handle the 'status' event from the server
    socket.on('clear-status', function(data) {
        document.getElementById('clearCorrectionsStatus').innerText = data.message;
    });
    
    let path = `/${classType}/clear_corrections`
    document.getElementById('clearCorrectionsButton').addEventListener('click', function() {
        document.getElementById('clearCorrectionsButton').style.display = 'none';
        fetch(path, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model })
        })
        .then(response => response.json())
        .then(data => {
            console.log(data);
            document.getElementById('correctionsPage').innerHTML = data["corrections"];
            document.getElementById('retrainPlots').innerHTML = data["retrain_plots"];
            document.getElementById('retrainButton').style.display = 'none';
            document.getElementById('retrainStatus').innerText = '';
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('clearCorrectionsStatus').innerText = "An error occurred.";
        });
    });
}