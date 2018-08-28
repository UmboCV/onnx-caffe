#!/usr/bin/env groovy
properties([
    disableConcurrentBuilds(),
    pipelineTriggers([
        issueCommentTrigger('.*test this please.*')
    ])
])

node {

    stage('Checkout') {
        checkout scm
    }
    def gitCommit = sh(returnStdout: true, script: 'git rev-parse HEAD').trim()
    def gitShortCommit = gitCommit.take(7)
    def gitBranch = env.BRANCH_NAME
    def gitCommitAuthor = sh(returnStdout: true, script: 'git show -s --format=%an').trim()
    def gitCommitMessage = sh(returnStdout: true, script: 'git show -s --format=%B').trim()


    def img_tag = gitBranch + '-' + gitShortCommit
    def ecr_repo = '774915305292.dkr.ecr.us-west-2.amazonaws.com/onnx-caffe'
    def build_docker_img = ecr_repo + ':' + img_tag

    slackSend color: '#439FE0', message: "${env.JOB_NAME} #${env.BUILD_NUMBER}: Pipeline started at commit $gitShortCommit by $gitCommitAuthor: $gitCommitMessage"

    // Test, for all branches
    try {
	    test: {
            stage('Test'){
                echo "Testing commit $gitShortCommit on branch $gitBranch as $build_docker_img"
                ansiColor('xterm') {
                    sh """
                        env IMAGE_NAME=$build_docker_img ./build.sh  && \
                        env IMAGE_NAME=$build_docker_img ./jenkins/test.sh
                    """
                }
            }
        }
    } catch (Exception e){
        throw (e)
    }
}
