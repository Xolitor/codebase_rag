function getUser(userId) {
    return {
        id: userId,
        name: "John Doe"
    };
}

function createUser(name) {
    return {
        id: Math.random(),
        name: name
    };
}

module.exports = {
    getUser,
    createUser
};