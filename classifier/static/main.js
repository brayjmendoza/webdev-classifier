function toggleSidebar() {
    /*
     * Toggle sidebar navigation visibility
     */
    const sidebar = document.getElementById('navLinks');
    const mainContent = document.getElementById('content');
    const isOpen = sidebar.style.width === '15%';

    if (isOpen) {
        sidebar.style.width = '0';
        mainContent.classList.remove('shifted');
        sidebar.classList.remove('extended');
    } else {
        sidebar.style.width = '15%';
        mainContent.classList.add('shifted');
        sidebar.classList.add('extended');
    }
}

function correction(classType, model) {
    /* Send corrections if prediction was wrong
     *
     * Parameters:
     * classType - the thing that was predicted (e.g. iris species)
     * model - the model used to predict (e.g. knn, dtree)
     */

    let path = `/${classType}/incorrect`;

    document.getElementById('correction').addEventListener('submit', function(e) {
        e.preventDefault(); // prevents reload
        const correction = document.getElementById('target').value;
        
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
            body: JSON.stringify({ correction, model })
        })
        .then(response => response.text())
        .then(data => {
            document.getElementById('correction').style.display='none';
            document.getElementById('feedback').innerText = 'Thanks for the feedback!';
            document.getElementById('correctionsPage').innerHTML = data;
            document.getElementById('retrainButton').style.display = 'block';

            // Restore corrections list state
            const tableContainer = document.getElementById('correctionsTableContainer');
            if (tableContainer) {
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
            }
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
                
                console.log(data)

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

function attachCorrectionListeners(classType, model) {
    /* Attach event listeners to the buttons in the corrections list
     * 
     * Parameters:
     * classType - the thing that was predicted (e.g. iris species)
     * model - the model used to predict (e.g. knn, dtree) 
     */

    function attachDropdownListener() {
        /*
         * Attach the event listener for dropdown functionality
        */
        const tableHeader = document.getElementById('correctionsHeader');
        const tableContainer = document.getElementById('correctionsTableContainer');
        const table = document.getElementById('correctionsTable');
        const caret = document.getElementById('caret');
        const clearButtonContainer = document.getElementById('clearCorrectionsContainer');

        if (tableHeader && tableContainer && caret) {
            // Dropdown functionality
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
                // Reattach the dropdown listener for new content
                attachDropdownListener();
                clearCorrections(classType, model);
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
     * Sends request to clear corrections table and corresponding data
     *
     * Parameters:
     * classType - the thing that was predicted (e.g. iris species)
     * model - the model used to predict (e.g. knn, dtree)
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