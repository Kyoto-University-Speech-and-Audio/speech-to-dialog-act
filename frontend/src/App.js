import React, { Component } from 'react';
import './App.css';
import {ReactMic} from "react-mic";
import {
  Button,
  FormControl, FormControlLabel, FormGroup,
  FormLabel, Grid, LinearProgress, Paper,
  Radio,
  RadioGroup, Typography
} from "@material-ui/core/es/index";
import {withStyles} from "@material-ui/core/es/styles/index";

const API_INFER = 'http://127.0.0.1:5000/infer';

class Recorder extends Component {
  constructor(props) {
    super(props);
    this.state = {
      isRecording: false,
      transcript: 'hello',
    }
  }

  startRecording = () => {
    this.setState({
      blobURL: "",
      isRecording: true
    });
  }

  stopRecording = () => {
    this.setState({
      isRecording: false
    });
  }

  onData(recordedBlob) {

  }

  onStop = (recordedBlob) => {
    this.setState({
      blobURL : recordedBlob.blobURL
    });
    this.props.onNewRecord(recordedBlob);
  }

  onStart = () => {

  }

  render() {
    return (
      <div>
        <ReactMic
          record={this.state.isRecording}
          className="sound-wave"
          onStart={this.onStart}
          onStop={this.onStop}
          onData={this.onData}
          visualSetting="sinewave"
          audioBitsPerSecond= {128000}
          strokeColor="#000000"
          backgroundColor="#ffffff" />
        <div>
          <audio ref="audioSource" controls="controls" src={this.state.blobURL}/>
        </div>
        {
          this.state.isRecording ?
          <Button
            onClick={this.stopRecording}
            disabled={this.props.isLoading}
            type="button"
            variant="outlined" color="primary">Stop
          </Button> :
          <Button
            onClick={this.startRecording} type="button"
            variant="outlined" color="primary">Start
          </Button>
        }
      </div>
    );
  }
}

const styles = theme => ({
  result: {
    ...theme.mixins.gutters(),
    paddingTop: theme.spacing.unit * 2,
    paddingBottom: theme.spacing.unit * 2,
    height: 200
  },
  resultGrid: {
    width: '70%'
  }
});

class App extends Component {
  state = {
    lang: 'en',
    transcript: "",
    isLoading: false
  };

  handleLanguageChange = event => {
    this.setState({
      lang: event.target.value
    });
  };

  onNewRecord = blob => {
    this.setState({
        blobURL : blob.blobURL
    })
    console.log('recordedBlob is: ', blob);

    const data = new FormData();
    const file = new File([blob.blob], "record.webm");
    data.append('lang', this.state.lang);
    data.append('file', file);
    console.log(data);
    this.setState({isLoading: true});

    fetch(API_INFER, {
      method: 'POST',
      body: data
    }).then(response => response.json())
      .then(data => {
        this.setState({
          transcript: data[0].text,
          isLoading: false
        })
      })
  };

  render() {
    const { classes } = this.props;

    return (
      <div className="App">
        <Grid container spacing={40} justify="center" direction="column" alignItems="center">
          <Grid item xs={12}>
            <Typography variant="h3" color="inherit" style={{ paddingTop: 40 }}>
              Speech Recognition
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <FormGroup row>
              <FormControl component="fieldset">
              <FormLabel component="legend">Language</FormLabel>
                <RadioGroup row
                  aria-label="Language"
                  name="language"
                  value={this.state.lang}
                  onChange={this.handleLanguageChange}
                >
                  <FormControlLabel value="en" control={<Radio />} label="English" />
                  <FormControlLabel value="ja" control={<Radio />} label="Japanese" />
                  <FormControlLabel value="vi" control={<Radio />} label="Vietnamese" />
                </RadioGroup>
              </FormControl>
            </FormGroup>
          </Grid>
          <Grid item xs={12}>
            <Recorder
              onNewRecord={this.onNewRecord}
              isLoading={this.state.isLoading}
            />
            { this.state.isLoading && <LinearProgress/> }
          </Grid>
          <Grid item xs={12} className={classes.resultGrid}>
            <Paper
              className={classes.result}
              elevation={5}
              square={true}>
              <Typography variant="h5">
                {this.state.transcript}
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      </div>
    );
  }
}

export default withStyles(styles)(App);
