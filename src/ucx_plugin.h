#include "p2p_plugin.h"
#include "ucp/api/ucp.h"

#define MAX_UCX_DEVS 32
#define NCCL_NET_UCX_MAX_RECVS 8

extern union ncclSocketAddress nccl_ucx_if_addr;
extern char if_name[MAX_IF_NAME_SIZE];

enum ncclUCXCommState {
  ncclUCXCommStateStart = 0,
  ncclUCXCommStateConnect = 1,
  ncclUCXCommStateAccept = 3,
};

struct ncclUCXCommStage {
  enum ncclUCXCommState state;
  uint8_t iteration;
  void *sock;
  void *comm;
};

struct ep_list {
  struct ncclSocket *sock;
  struct ep_list *next;
};

/**
 * Connection descriptor. Used to store all opened connections.
 */
typedef struct nccl_ucx_worker {
  ucp_worker_h worker; /* ucp worker associated with ctx */
  ucp_context_h ctx;   /* ucp_context bounded to specific device */
  struct ep_list *eps; /* oob conection to all endpoints that were opened on
                          this worker */

  int count;        /* number of connections that uses this worker */
  int dev;          /* Managed device */
  pthread_t thread; /* Owner thread */

  struct nccl_ucx_worker *next;
} nccl_ucx_worker_t;

/**
 * Listen handle that is sent from receiver to sender through OOB connection
 */
typedef struct ucx_listen_handle {
  union ncclSocketAddress connectAddr; /* reciever socket address */
  uint64_t magic;                      /* random number to help debugging */
  ucp_tag_t tag; /* tag that is used to distiguish data that was sent to
                    this reciever. Required when shared worker is used. */
  struct ncclUCXCommStage stage;
} ucx_listen_handle_t;

/**
 * Listen commincator for UCX plugin.
 */
typedef struct ucx_listen_comm {
  int dev;                /* device number in ncclIbDevs which will
                           * be used to recieve data */
  struct ncclSocket sock; /* socket for OOB connection */
  ucp_context_h ctx;      /* ucp_context associated with specific device dev */
  nccl_ucx_worker_t *ucx_worker; /* ucx_worker created on ctx, worker can be
                           shared between multiple connections */
  ucp_tag_t tag; /* tag that is used to distiguish data that was sent to
                    this reciever. Required when shared worker is used.*/
  struct ncclUCXCommStage stage;
} ucx_listen_comm_t;

typedef struct connect_msg {
  size_t addr_len;
} connect_msg_t;

typedef struct ucx_gpu_flush {
  int enabled;
  int hostMem;
  ucp_ep_h flush_ep;
} ucx_gpu_flush_t;

struct ucx_comm;

/**
 * Batch of UCX Requests from NCCL perspective
 */
typedef struct ucx_request {
  struct ucx_request *next; /* Next request in the free list */
  struct ucx_comm *comm;    /* Owning communicator */
  ucp_worker_h worker;      /* Worker for all requests */
  int pending;              /* How many requests are still pending */
  int count;                /* How many requests are contained */
  int size[NCCL_NET_UCX_MAX_RECVS];
} ucx_request_t;

/**
 * Common data member for ucx_comm for send and receive
 * Used to map/unmap memory in nccl_ucx_regmr/nccl_ucx_deregmr
 */
typedef struct ucx_ctx {
  ucp_context_h ucp_ctx;
  ucx_gpu_flush_t gpuFlush;
} ucx_ctx_t;

/**
 * Sender and Receiver communicator
 */
typedef struct ucx_comm {
  ucp_context_h ctx;             /* ucp_context bounded to specific device */
  ucx_gpu_flush_t gpuFlush;      /* flushing handle */
  nccl_ucx_worker_t *ucx_worker; /* ucp worker associated with ctx */
  ucp_ep_h ep;                   /* ucp endpoint created on worker */
  ucp_tag_t tag;          /* datapath tag to filter out message that are not
                             belong to this connnection */
  ucp_tag_t ctag;         /* controlpath tag to filter out message that are not
                             belong to this connnection */
  struct ncclSocket sock; /* socket for OOB connection */
  int ready; /* indicates that receive communicator is fully initialized */
  ucx_request_t reqs[MAX_REQUESTS]; /* max inflight requests */
  ucx_request_t *free_req;          /* first request available */
  connect_msg_t *msg; /* message to establish reverse connection */
  void *connect_req;  /* msg request */
} ucx_comm_t;

typedef struct nccl_bxi_properties {
  uint64_t guid;
  char name[MAXNAMESIZE];
  char *pciPath;
  int hmem_support;
  /** support dmabuf interface */
  int dmabuf_support;
  /** Port number */
  int port_number;
  /** Port speed in Mbps */
  int port_speed;
  /** Port latency */
  float latency;
  /** Maximum number of comms supported */
  unsigned int max_communicators;
  /** Maximum number of grouped receives */
  unsigned int max_group_receives;
  /** regMr is global if is not tied to a particular comm **/
  int regIsGlobal;
  /** Maximum size of buffer supported to be transfered via
   * RMA write inline operation **/
  size_t max_write_inline_size;
  /** Maximum size of the memory region remote access key in bytes **/
  size_t max_mr_key_size;
  /** Indicator whether RMA operations of NCCL Net API are supported **/
  int rma_supported;
  /** Max transfer size for point-to-point operations **/
  size_t max_p2p_bytes;
  /** Max transfer size for collective operations **/
  size_t max_coll_bytes;
} nccl_bxi_properties_t;

typedef struct nccl_bxi_dev {
  int device;
  nccl_bxi_properties_t props;
} nccl_bxi_dev_t;

extern int nccl_nb_bxi_devs;
extern nccl_bxi_dev_t nccl_bxi_devs[MAX_UCX_DEVS];

typedef struct ucx_mhandle {
  ucp_mem_h ucp_memh;
  ucp_rkey_h rkey;
  int mem_type;
} ucx_mhandle_t;

ncclResult_t nccl_ucx_param_init();

ncclResult_t ucx_init_context(ucp_context_h *ctx, int dev);
ncclResult_t ucx_init_worker(ucp_context_h ctx, ucp_worker_h *worker);
ncclResult_t ucx_get_ctx_and_worker(int dev, ucp_context_h *ctx,
                                    nccl_ucx_worker_t **ucx_worker,
                                    ucp_tag_t *newtag);
ncclResult_t ucx_worker_get_netaddress(ucp_worker_h worker,
                                       ucp_address_t **address,
                                       size_t *address_length);
ncclResult_t nccl_ucx_free_worker(nccl_ucx_worker_t *ucx_worker);
ncclResult_t nccl_ucx_add_ep(nccl_ucx_worker_t *ucx_worker,
                             struct ncclSocket *sock);
void ucx_request_init(ucx_comm_t *comm);
ucx_request_t *ucx_request_get(ucx_comm_t *comm);
void ucx_request_release(ucx_request_t *req);
void ucx_request_add(ucx_request_t *req, int size);

// OOB
ncclResult_t nccl_ucx_listen(void *ctx, int dev, void *handle,
                             void **listen_comm);
ncclResult_t nccl_ucx_connect(void *ctx, int dev, void *handle,
                              void **send_comm,
                              ncclNetDeviceHandle_t **sendDevComm);
ncclResult_t nccl_ucx_accept(void *listen_comm, void **recv_comm,
                             ncclNetDeviceHandle_v7_t **recvDevComm);

// Mem reg
ncclResult_t nccl_ucx_regmr(void *comm, void *data, size_t size, int type,
                            void **mhandle);
ncclResult_t nccl_ucx_deregmr(void *comm, void *mhandle);
ncclResult_t nccl_ucx_regmr_dmabuf(void *comm, void *data, size_t size,
                                   int type, uint64_t offset, int fd,
                                   void **mhandle);

// Send/recv
ncclResult_t ucx_send_check(ucx_comm_t *comm);
void ucx_recv_set_ready(ucx_comm_t *comm);
void check_handler(void *request, ucs_status_t status, void *user_data);
ncclResult_t ucx_recv_check(ucx_comm_t *comm);
ucp_tag_t nccl_ucx_ucp_tag(ucp_tag_t comm_tag, uint64_t tag);
ncclResult_t nccl_ucx_isend(void *send_comm, void *data, size_t size, int tag,
                            void *mhandle, void *phandle, void **request);
ncclResult_t nccl_ucx_irecv(void *recv_comm, int n, void **data, size_t *sizes,
                            int *tags, void **mhandle, void **phandles,
                            void **request);

// Sync
ncclResult_t nccl_ucx_iflush(void *recv_comm, int n, void **data, int *sizes,
                             void **mhandle, void **request);
ncclResult_t nccl_ucx_test(void *request, int *done, int *size);
void wait_close(ucp_worker_h worker, void *ucp_req);
ncclResult_t nccl_ucx_close_send(void *send_comm);
ncclResult_t nccl_ucx_close_recv(void *recv_comm);
ncclResult_t nccl_ucx_close_listen(void *listen_comm);
ncclResult_t nccl_ucx_finalize(void *ctx);

ncclResult_t nccl_ucx_setNetAttr(void *ctx, ncclNetAttr_t *netAttr);
